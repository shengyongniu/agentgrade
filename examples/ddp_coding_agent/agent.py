"""A more complete Coder -> Critic pipeline that PASSES all checks.

Use this alongside ``examples/simple_agent`` to compare a passing run against
a failing one. Fully deterministic and offline.
"""

from __future__ import annotations

from agentgrade.integrations import TraceRecorder
from agentgrade.trace import AgentTrace


def _coder_agent(task: str) -> str:
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "from torch.utils.data import DataLoader\n"
        "from torch.utils.data.distributed import DistributedSampler\n\n"
        "def train():\n"
        "    torch.distributed.init_process_group(backend='nccl')\n"
        "    local_rank = torch.distributed.get_rank()\n"
        "    model = nn.Linear(10, 10).cuda(local_rank)\n"
        "    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])\n"
        "    sampler = DistributedSampler(dataset)\n"
        "    loader = DataLoader(dataset, batch_size=32, sampler=sampler)\n"
        "    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n"
        "    for batch in loader:\n"
        "        optimizer.zero_grad()\n"
        "        loss = model(batch).sum()\n"
        "        loss.backward()\n"
        "        optimizer.step()\n\n"
        "if __name__ == '__main__':\n"
        "    train()\n"
    )


def _critic_agent(task: str, draft: str) -> str:
    launch = (
        "\n# Launch with:\n"
        "# torchrun --nproc_per_node=4 train.py\n"
    )
    return draft + launch


def run_agent(task: str) -> tuple[str, AgentTrace]:
    rec = TraceRecorder(test_name="ddp_training_script_complete")

    draft = _coder_agent(task)
    rec.step(
        "CoderAgent",
        input=task,
        output=draft,
        tool_name="codegen",
        tool_input={"task": task},
        tool_output="drafted complete DDP training loop",
        latency_ms=140,
        cost_usd=0.005,
    )

    reviewed = _critic_agent(task, draft)
    rec.step(
        "CriticAgent",
        input=draft,
        output=reviewed,
        tool_name="review",
        tool_input={"draft_len": len(draft)},
        tool_output="added torchrun launch command",
        latency_ms=80,
        cost_usd=0.003,
    )

    return reviewed, rec.finalize(final_output=reviewed)
