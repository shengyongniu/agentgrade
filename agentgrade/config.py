"""Configuration schema and loading for agentgrade."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class CheckConfig(BaseModel):
    """A single check/evaluator configuration.

    Extra keys (e.g. ``value``, ``seconds``, ``usd``, ``pattern``, ``path``)
    are kept so evaluators can read whatever parameters they need.
    """

    model_config = {"extra": "allow"}

    type: str
    weight: float = 1.0
    agent_name: str | None = None


class TestConfig(BaseModel):
    name: str
    input: str = ""
    checks: list[CheckConfig] = Field(default_factory=list)


class AgentConfig(BaseModel):
    type: str = "python"
    entrypoint: str


class Settings(BaseModel):
    fail_below_reward: float = 0.75
    output_dir: str = ".agentgrade"
    replay: bool = False
    fixtures_dir: str | None = None
    """Directory holding recorded trace fixtures.

    When unset, defaults to ``<output_dir>/fixtures`` via
    :meth:`resolved_fixtures_dir`.
    """

    def resolved_fixtures_dir(self) -> Path:
        """Return the fixtures directory, defaulting to ``<output_dir>/fixtures``."""

        if self.fixtures_dir:
            return Path(self.fixtures_dir)
        return Path(self.output_dir) / "fixtures"


class AgentGradeConfig(BaseModel):
    agent: AgentConfig
    tests: list[TestConfig] = Field(default_factory=list)
    settings: Settings = Field(default_factory=Settings)
    plugins: list[str] = Field(default_factory=list)
    """Plugin entrypoints to import before running checks.

    Each entry is a ``module.path:function`` string (or a bare ``module.path``)
    that, when imported/called, registers custom evaluators via
    :func:`agentgrade.evaluators.register_evaluator`.
    """


DEFAULT_CONFIG_FILENAME = "agentgrade.yaml"


def load_config(path: str | Path) -> AgentGradeConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    return AgentGradeConfig.model_validate(data)


EXAMPLE_CONFIG = """\
agent:
  type: python
  entrypoint: examples.simple_agent.agent:run_agent

tests:
  - name: ddp_training_script
    input: "Write a PyTorch DDP training script."
    checks:
      - type: contains
        value: "DistributedSampler"
        weight: 0.2
        agent_name: CoderAgent
      - type: contains
        value: "DistributedDataParallel"
        weight: 0.2
        agent_name: CoderAgent
      - type: contains
        value: "init_process_group"
        weight: 0.2
        agent_name: CoderAgent
      - type: regex
        value: "torchrun|python -m torch.distributed.run"
        weight: 0.2
        agent_name: CriticAgent
      - type: max_latency
        seconds: 30
        weight: 0.1
      - type: max_cost
        usd: 1.0
        weight: 0.1

settings:
  fail_below_reward: 0.75
  output_dir: ".agentgrade"
"""


def write_example_config(path: str | Path) -> Path:
    path = Path(path)
    path.write_text(EXAMPLE_CONFIG)
    return path
