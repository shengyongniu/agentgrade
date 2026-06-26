"""agentgrade command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from .config import DEFAULT_CONFIG_FILENAME, load_config, write_example_config
from .improve import build_patch_markdown
from .report import render_terminal, save_json, save_markdown
from .runner import record_suite, run_suite

app = typer.Typer(
    add_completion=False,
    help="agentgrade — pytest for multi-agent systems.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init(
    path: str = typer.Option(DEFAULT_CONFIG_FILENAME, help="Config file to create."),
) -> None:
    """Create an example agentgrade.yaml if none exists."""

    target = Path(path)
    if target.exists():
        console.print(f"[yellow]{target} already exists; leaving it untouched.[/]")
        raise typer.Exit(code=0)
    write_example_config(target)
    console.print(f"[green]Created {target}[/]. Run [bold]agentgrade test[/] to try it.")


@app.command()
def test(
    config: str = typer.Option(DEFAULT_CONFIG_FILENAME, help="Path to agentgrade.yaml."),
    replay: bool = typer.Option(
        False, "--replay", help="Replay recorded fixtures instead of calling the agent."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Print only the JSON results to stdout (machine mode)."
    ),
) -> None:
    """Run the configured agent tests and report results."""

    cfg_path = Path(config)
    if not cfg_path.exists():
        console.print(f"[red]Config not found:[/] {cfg_path}. Run [bold]agentgrade init[/] first.")
        raise typer.Exit(code=2)

    cfg = load_config(cfg_path)
    results = run_suite(cfg, replay=replay or None)

    json_path = save_json(results, cfg.settings)
    md_path = save_markdown(results, cfg.settings)

    if json_out:
        typer.echo(json_path.read_text())
    else:
        render_terminal(results, cfg.settings, console=console)
        console.print(f"[dim]JSON:[/] {json_path}")
        console.print(f"[dim]Markdown:[/] {md_path}")

    if any(not r.passed for r in results):
        raise typer.Exit(code=1)


@app.command()
def record(
    config: str = typer.Option(DEFAULT_CONFIG_FILENAME, help="Path to agentgrade.yaml."),
) -> None:
    """Run the real agent once and save each test's trace as a replay fixture."""

    cfg_path = Path(config)
    if not cfg_path.exists():
        console.print(f"[red]Config not found:[/] {cfg_path}. Run [bold]agentgrade init[/] first.")
        raise typer.Exit(code=2)

    cfg = load_config(cfg_path)
    paths = record_suite(cfg)

    console.print(f"[green]Recorded {len(paths)} fixture(s):[/]")
    for path in paths:
        console.print(f"  {path}")
    console.print(
        "Run [bold]agentgrade test --replay[/] for deterministic, offline checks."
    )


@app.command()
def report(
    config: str = typer.Option(DEFAULT_CONFIG_FILENAME, help="Path to agentgrade.yaml."),
    show: bool = typer.Option(True, help="Print a terminal summary of the latest run."),
) -> None:
    """Print the path to the latest report (and optionally a summary)."""

    cfg = load_config(config) if Path(config).exists() else None
    output_dir = Path(cfg.settings.output_dir) if cfg else Path(".agentgrade")
    md_path = output_dir / "reports" / "latest.md"

    if not md_path.exists():
        console.print("[red]No report found.[/] Run [bold]agentgrade test[/] first.")
        raise typer.Exit(code=2)

    console.print(f"[green]Latest report:[/] {md_path}")
    if show:
        console.print(md_path.read_text())


def _group_patches_by_agent(patches: list) -> dict[str, list[str]]:
    """Group ``[Agent] item`` patch strings by agent, defensively.

    Skips non-string entries, tolerates a missing/empty agent label (falls back
    to ``"Agent"``), and never raises on malformed input so a corrupt results
    file can't crash ``agentgrade improve``.
    """

    by_agent: dict[str, list[str]] = {}
    for patch in patches:
        if not isinstance(patch, str):
            continue
        agent = "Agent"
        item = patch
        if patch.startswith("[") and "]" in patch:
            close = patch.index("]")
            candidate = patch[1:close].strip()
            if candidate:
                agent = candidate
            item = patch[close + 1 :].strip()
        by_agent.setdefault(agent, []).append(item)
    return by_agent


@app.command()
def improve(
    config: str = typer.Option(DEFAULT_CONFIG_FILENAME, help="Path to agentgrade.yaml."),
    suggest: bool = typer.Option(False, "--suggest", help="Generate a prompt patch suggestion."),
) -> None:
    """Suggest a deterministic prompt patch from the latest failed run."""

    if not suggest:
        console.print("Pass [bold]--suggest[/] to generate a prompt patch from the last run.")
        raise typer.Exit(code=0)

    cfg = load_config(config) if Path(config).exists() else None
    output_dir = Path(cfg.settings.output_dir) if cfg else Path(".agentgrade")
    results_path = output_dir / "results" / "latest.json"

    if not results_path.exists():
        console.print("[red]No results found.[/] Run [bold]agentgrade test[/] first.")
        raise typer.Exit(code=2)

    results = json.loads(results_path.read_text())
    failed = [r for r in results if not r["passed"]]
    if not failed:
        console.print("[green]No failed tests — nothing to patch.[/]")
        raise typer.Exit(code=0)

    sections: list[str] = []
    for result in failed:
        patches = result.get("suggested_patches", [])
        if not patches:
            continue
        by_agent = _group_patches_by_agent(patches)

        sections.append(f"# Suggested prompt patch for `{result['name']}`")
        sections.append("")
        sections.append("Append the following checklist items to the responsible agent prompts:")
        sections.append("")
        for agent, items in by_agent.items():
            sections.append(f"## {agent}")
            sections.append("")
            sections.append("```diff")
            sections.append("  Before returning your answer, verify:")
            for item in items:
                sections.append(f"+ - [ ] {item}")
            sections.append("```")
            sections.append("")

    patch_md = "\n".join(sections) + "\n"
    patch_path = output_dir / "reports" / "suggested_patch.md"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch_md)

    console.print(patch_md)
    console.print(f"[green]Wrote[/] {patch_path}")


if __name__ == "__main__":
    app()
