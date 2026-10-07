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
"""mini-swe-agent style agent loop: a bash tool in a per-trajectory sandbox.

The loop reuses :class:`ToolAgentLoop` for generation and tool calls and adds the
sandbox lifecycle around it:

1. open a sandbox from ``extra_info["sandbox"]`` and run its setup commands;
2. let the model call the ``bash`` tool until it answers without a tool call
   (or hits the turn/length limits);
3. run the spec's ``eval_command`` in the same sandbox and store the result in
   ``extra_fields["sandbox_eval"]``, which reaches the reward function as
   ``extra_info["sandbox_eval"]`` (see ``verl.utils.reward_score.sandbox_eval``);
4. remove the sandbox, also when the episode fails.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.tools.bash_tool import BashTool, SandboxSpec

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

EVAL_OUTPUT_TAIL_CHARS = 2000


@register("mini_swe_agent")
class MiniSweAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bash_tools = [tool for tool in self.tools.values() if isinstance(tool, BashTool)]
        if not bash_tools:
            raise ValueError(
                "mini_swe_agent needs a BashTool: add verl.tools.bash_tool.BashTool to "
                "actor_rollout_ref.rollout.multi_turn.tool_config_path"
            )
        self.bash_tool: BashTool = bash_tools[0]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info") or {}
        spec = SandboxSpec.from_value(extra_info.get("sandbox"))
        # Not the request id: under full_determinism request ids repeat across samples.
        self._sandbox_session_id = uuid4().hex

        await self.bash_tool.open_session(self._sandbox_session_id, spec)
        try:
            output = await super().run(sampling_params, **kwargs)
            output.extra_fields["sandbox_eval"] = await self._evaluate(spec)
            return output
        finally:
            await self.bash_tool.close_session(self._sandbox_session_id)

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        # BashTool.execute finds the trajectory's sandbox through agent_data.
        agent_data.sandbox_session_id = self._sandbox_session_id
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _evaluate(self, spec: SandboxSpec) -> dict[str, Any]:
        result = {
            "evaluated": False,
            "returncode": None,
            "timed_out": False,
            "output": "",
            "error": None,
            **self.bash_tool.session_stats(self._sandbox_session_id),
        }
        if not spec.eval_command:
            return result
        timeout = spec.eval_timeout if spec.eval_timeout is not None else self.bash_tool.eval_timeout
        try:
            eval_result = await self.bash_tool.run_command(self._sandbox_session_id, spec.eval_command, timeout)
        except Exception as e:
            logger.warning(f"Sandbox evaluation failed: {e}")
            result["error"] = str(e)
            return result
        result.update(
            evaluated=True,
            returncode=eval_result.returncode,
            timed_out=eval_result.timed_out,
            output=eval_result.output[-EVAL_OUTPUT_TAIL_CHARS:],
        )
        return result
