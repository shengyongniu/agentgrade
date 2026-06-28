# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-28

Initial public release.

### Added

- `agentgrade` CLI with `init`, `test`, `record`, `report`, and `improve` commands.
- Regression testing for multi-agent workflows with weighted-reward checks
  (`contains`, `not_contains`, `regex`, `exact_match`, `max_latency`, `max_cost`,
  `python_import_check`, `unit_tests`).
- Credit assignment that pins each failed check on the responsible agent, with
  inference when no `agent_name` is annotated.
- Deterministic prompt-patch suggestions from the last failed run.
- Record/replay for deterministic, offline CI runs.
- Plugin API for custom evaluators and patch suggesters (decorator, registry, and
  `agentgrade.evaluators` entry-point group).
- LangGraph adapter (`agentgrade[langgraph]` extra).
- Bundled GitHub Action and example agent pipelines.

[0.1.0]: https://github.com/shengyongniu/agentgrade/releases/tag/v0.1.0
