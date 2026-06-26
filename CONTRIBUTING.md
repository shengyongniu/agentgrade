# Contributing to agentgrade

Thanks for your interest in improving agentgrade! This project is a local-first,
deterministic, offline CLI — "pytest for multi-agent systems" — and contributions that
keep it that way are very welcome.

## Development environment

agentgrade targets Python 3.10+. Set up a virtual environment and install the package in
editable mode with the dev extras:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
```

To work on the LangGraph adapter against a real graph, also install the optional extra:

```bash
pip install -e ".[dev,langgraph]"
```

## Running the examples

The bundled examples are fully deterministic and need no API keys:

```bash
# A failing Coder -> Critic pipeline (reward 0.60, exit 1) with named-agent credit.
agentgrade test --config examples/simple_agent/agentgrade.yaml

# A passing, complete pipeline (reward 1.00).
agentgrade test --config examples/ddp_coding_agent/agentgrade.yaml

# Inferred credit (no agent_name annotations) + a config-loaded custom evaluator.
agentgrade test --config examples/inferred_agent/agentgrade.yaml

# Record once, then replay deterministically/offline.
agentgrade record --config examples/simple_agent/agentgrade.yaml
agentgrade test --config examples/simple_agent/agentgrade.yaml --replay

# Generate a deterministic prompt patch from the last failed run.
agentgrade improve --config examples/simple_agent/agentgrade.yaml --suggest
```

## Adding a custom evaluator (plugin API)

New check types do not require forking. Register an evaluator with the `@evaluator`
decorator (or `register_evaluator(name, fn)`):

```python
from agentgrade.evaluators import evaluator
from agentgrade.trace import EvaluationResult

@evaluator("min_length")
def eval_min_length(output, trace, check):
    minimum = int(getattr(check, "min_chars", 0) or 0)
    passed = len(output) >= minimum
    return EvaluationResult(
        check_name=f"min_length:{minimum}",
        passed=passed,
        score=1.0 if passed else 0.0,
        weight=check.weight,
        message="ok" if passed else "too short",
    )
```

Load it declaratively via the config `plugins:` key (`module.path:function` or bare
`module.path`), or, for installed packages, advertise it under the
`agentgrade.evaluators` entry-point group. See `examples/inferred_agent/plugins.py` for a
working example. To teach `agentgrade improve` how to phrase a patch for your check type,
register a `register_patch_suggester` hook.

## Adding a framework adapter

Adapters live in `agentgrade/integrations/` and convert a framework's run into
agentgrade's `(final_output, AgentTrace)` shape — one `AgentStep` per named agent/tool
step. Use `agentgrade/integrations/langgraph.py` as a template:

- **Duck-type** framework objects (access fields via `getattr`, recognise message types by
  `type(obj).__name__`) so the module imports cleanly even without the framework
  installed — never add a hard top-level import of an optional dependency.
- Add the dependency as an optional extra in `pyproject.toml`
  (`[project.optional-dependencies]`), document `pip install agentgrade[<name>]`, and
  re-export the entrypoint from `agentgrade/integrations/__init__.py`.
- Ship an **offline** unit test that drives a fake/stub object (no network, no API keys),
  mirroring `tests/test_langgraph_adapter.py`.

## Code style

- Fully typed; we use modern type hints and `from __future__ import annotations`.
- Data models are [Pydantic v2](https://docs.pydantic.dev/latest/) `BaseModel`s.
- Keep it deterministic and offline: no network calls, no required API keys, no LLM calls
  in core logic.
- Comments explain non-obvious intent only — avoid narrating what the code does.

## Pull requests

1. Fork and create a topic branch.
2. Keep changes focused; add or update tests for any behavior change.
3. Make sure `pytest -q` is green and the bundled example outputs are unchanged
   (reward/credit numbers are part of the contract).
4. Describe the motivation ("why") in the PR description and link any related issue.

By contributing you agree that your contributions are licensed under the project's
[MIT License](LICENSE).
