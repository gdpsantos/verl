# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end CPU test of the mini_swe_agent loop with a scripted model and the local sandbox."""

import json
import os
from typing import Any

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
from verl.experimental.agent_loop.mini_swe_agent_loop import MiniSweAgentLoop
from verl.tools.bash_tool import BashTool
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.reward_score import default_compute_score
from verl.workers.rollout.replica import TokenOutput

TOKENIZER = os.environ.get("VERL_TEST_TOKENIZER", "Qwen/Qwen2.5-0.5B-Instruct")

BUGGY = "def sum_to(n):\n    return sum(range(1, n))\n"
TESTS = (
    "import unittest\nfrom utils import sum_to\n\n"
    "class T(unittest.TestCase):\n    def test(self):\n        self.assertEqual(sum_to(5), 15)\n"
)
SANDBOX = {
    "setup_commands": [
        f"cat > utils.py <<'EOF'\n{BUGGY}EOF",
        f"cat > test_utils.py <<'EOF'\n{TESTS}EOF",
    ],
    "eval_command": f"cat > test_utils.py <<'EOF'\n{TESTS}EOF\npython3 -m unittest -q test_utils",
}


def _tool_call(command: str) -> str:
    return f"<tool_call>\n{json.dumps({'name': 'bash', 'arguments': {'command': command}})}\n</tool_call>"


class _ScriptedServer:
    """Plays back assistant turns and records the observations the model would see."""

    def __init__(self, tokenizer, turns: list[str]):
        self.tokenizer = tokenizer
        self.turns = list(turns)
        self.prompts: list[str] = []

    async def generate(self, request_id: str, *, prompt_ids: list[int], sampling_params: dict, **kwargs) -> TokenOutput:
        self.prompts.append(self.tokenizer.decode(prompt_ids))
        text = self.turns.pop(0) + "<|im_end|>"
        return TokenOutput(token_ids=self.tokenizer.encode(text, add_special_tokens=False))


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except OSError as e:
        pytest.skip(f"tokenizer {TOKENIZER} unavailable: {e}")


def _make_loop(tokenizer, server, tools: list[Any]) -> MiniSweAgentLoop:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 2048,
                    "response_length": 4096,
                    "full_determinism": False,
                    "multi_turn": {
                        "max_user_turns": None,
                        "max_assistant_turns": 6,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 12000,
                        "tool_response_truncate_side": "middle",
                        "format": "hermes",
                    },
                }
            },
            "data": {},
        }
    )
    return MiniSweAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        tools=ToolListWrap(tools),
    )


async def _run(tokenizer, turns: list[str], tool: BashTool):
    server = _ScriptedServer(tokenizer, turns)
    loop = _make_loop(tokenizer, server, [tool])
    output = await loop.run(
        {"temperature": 1.0},
        raw_prompt=[{"role": "user", "content": "Fix the bug in utils.py so the tests pass."}],
        extra_info={"sandbox": SANDBOX},
        tools_kwargs={},
    )
    return output, server


@pytest.mark.asyncio
async def test_solved_episode_passes_eval_and_cleans_up(tokenizer):
    tool = BashTool(config={"type": "native", "backend": "local"})
    turns = [
        _tool_call("python3 -m unittest -q test_utils"),
        _tool_call("sed -i 's/range(1, n)/range(1, n + 1)/' utils.py && cat utils.py"),
        "I fixed the off-by-one error in sum_to.",
    ]
    output, server = await _run(tokenizer, turns, tool)

    assert tool._sessions == {}
    result = output.extra_fields["sandbox_eval"]
    assert result["evaluated"] and result["returncode"] == 0 and not result["timed_out"]
    assert result["num_commands"] == 2
    assert output.num_turns == 6  # user + 3 assistant + 2 tool

    # The model saw the failing test and the edited file as tool observations.
    assert "<returncode>1</returncode>" in server.prompts[1] and "FAILED" in server.prompts[1]
    assert "range(1, n + 1)" in server.prompts[2]
    # Observation tokens are masked out of the loss, model tokens are not.
    assert 0 in output.response_mask and output.response_mask[0] == 1

    score = default_compute_score("sandbox_eval/toy", "", "", extra_info={"sandbox_eval": result})
    assert score["score"] == 1.0 and score["num_commands"] == 2.0


@pytest.mark.asyncio
async def test_unsolved_episode_and_test_tampering_score_zero(tokenizer):
    tool = BashTool(config={"type": "native", "backend": "local"})
    # Overwriting the tests does not help: eval rewrites them before running.
    turns = [_tool_call("echo 'import unittest' > test_utils.py"), "Done."]
    output, _ = await _run(tokenizer, turns, tool)

    result = output.extra_fields["sandbox_eval"]
    assert result["evaluated"] and result["returncode"] != 0 and "FAILED" in result["output"]
    score = default_compute_score("sandbox_eval", "", "", extra_info={"sandbox_eval": result})
    assert score == {"score": 0.0, "acc": 0.0, "eval_timed_out": 0.0, "sandbox_error": 0.0, "num_commands": 1.0}
    assert tool._sessions == {}


@pytest.mark.asyncio
async def test_session_closed_when_generation_fails(tokenizer):
    tool = BashTool(config={"type": "native", "backend": "local"})
    with pytest.raises(IndexError):
        await _run(tokenizer, [_tool_call("ls")], tool)  # server runs out of turns
    assert tool._sessions == {}


def test_requires_bash_tool(tokenizer):
    with pytest.raises(ValueError, match="needs a BashTool"):
        _make_loop(tokenizer, server=None, tools=[])
