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
"""Reward from a sandbox evaluation run by the mini_swe_agent agent loop.

The agent loop runs the sample's ``eval_command`` in the trajectory's sandbox after
the episode and stores the outcome in ``extra_info["sandbox_eval"]``. The task is
solved when that command exits with code 0 within its timeout.
"""

from typing import Any, Optional


def compute_score(solution_str: str, ground_truth: Any, extra_info: Optional[dict] = None, **kwargs) -> dict:
    """Score 1.0 if the sandbox evaluation command passed, else 0.0.

    Always returns the same keys so per-sample reward_extra_info arrays stay rectangular.
    """
    result = (extra_info or {}).get("sandbox_eval") or {}
    passed = bool(result.get("evaluated")) and result.get("returncode") == 0 and not result.get("timed_out")
    score = 1.0 if passed else 0.0
    return {
        "score": score,
        "acc": score,
        "eval_timed_out": float(bool(result.get("timed_out"))),
        "sandbox_error": float(bool(result.get("error")) or not result.get("evaluated", False)),
        "num_commands": float(result.get("num_commands") or 0),
    }
