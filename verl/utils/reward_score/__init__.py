# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def _tool_execution_bonus(solution_str: str, bonus: float = 0.2) -> float:
    """Three-gate tool-execution bonus (ToolRL/ToolRLA principle).

    A small shaping reward, awarded once per trajectory, that survives an
    adversarial optimizer: the model only earns it by actually calling a tool and
    getting a usable result back. All three gates must pass:

    1. At least one syntactically valid <tool_call> (valid JSON with 'name' and
       'arguments').
    2. Evidence the harness actually executed it: a tool message immediately
       follows the call. The agent loop always inserts the tool response right
       after </tool_call>, rendered by the chat template as
       "<|im_start|>tool\\n{output}<|im_end|>". With skip_special_tokens=True the
       special markers are stripped, leaving "</tool_call>tool\\n{output}" — so we
       require the "tool\\n" role label glued to </tool_call>. This is what makes
       the gate hard to fake: a model cannot synthesize that label itself, the
       harness adds it only on a real execution.
    3. The tool output is non-empty and not an error.

    Tool output may be a plain string (python_interpreter_server backend),
    JSON {"stdout": ..., "stderr": ...} or {"result": ...} (the shapes returned
    by verl.tools.mcp_base_tool.MCPBaseTool, including its error paths), and may
    optionally be wrapped in <tool_response>...</tool_response> tags.
    """
    import json
    import re

    # The "Search ..."/"No search"/"Unknown API" alternatives are the failure strings of
    # the search tool (verl/tools/search_tool.py, verl/tools/utils/search_r1_like_utils.py),
    # so a down retriever earns no bonus.
    error_pattern = re.compile(
        r"^(Error[:\s]|Tool (call|execution) failed|Connection failed|An unexpected error occurred"
        r"|Search request failed|Search error|Search execution failed|No search results found"
        r"|Unknown API state)"
    )
    # XML tool-call format (qwen3_coder parser): <tool_call><function=NAME>...</function></tool_call>
    xml_call = re.compile(r"<function=\s*[^>\s]+\s*>.*?</function>", re.DOTALL)

    # Gate 1: at least one syntactically valid tool call.
    has_valid_call = False
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL):
        if xml_call.search(m.group(1)):
            has_valid_call = True
            break
        try:
            obj = json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            has_valid_call = True
            break
    if not has_valid_call:
        return 0.0

    # Gates 2 & 3: a tool message must follow a call, and at least one such output
    # must be non-empty and non-error. Award the bonus on the first clean execution.
    # Two renderings of the harness-inserted tool turn are accepted:
    #   "</tool_call>...tool\n{output}"                     (tool role, e.g. AMALIA template)
    #   "</tool_call>...user\n<tool_response>{output}..."   (Qwen3-style: results in a user turn)
    # The lookahead keeps <tool_response> in the payload so it is unwrapped below.
    for lbl in re.finditer(r"</tool_call>\s*(?:tool\n|user\n(?=\s*<tool_response>))", solution_str):
        payload = solution_str[lbl.end():]
        # Cut at the next turn's role label (newline-anchored so the same words
        # appearing inside the tool output are not treated as a boundary).
        boundary = re.search(r"\n\s*(?:assistant|user|tool)\b", payload)
        if boundary:
            payload = payload[: boundary.start()]
        # Unwrap optional <tool_response>...</tool_response>.
        tr = re.search(r"<tool_response>(.*?)</tool_response>", payload, re.DOTALL)
        if tr:
            payload = tr.group(1)
        payload = payload.strip()
        if not payload:
            continue

        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            parsed = None

        if isinstance(parsed, dict):
            text = (parsed.get("stdout") or parsed.get("result") or "").strip()
            if text and not error_pattern.match(text):
                return bonus
        else:
            if not error_pattern.match(payload) and "produced no output" not in payload:
                return bonus

    return 0.0


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    elif data_source in [
        "searchR1_ptpt",
        "searchR1_ptpt_singlehop",
        "searchR1_ptpt_multihop",
    ]:
        from . import search_r1_like_qa_em_ptpt

        res = search_r1_like_qa_em_ptpt.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        score = res
    elif isinstance(res, int | float | bool):
        score = float(res)
    else:
        score = float(res[0])

    # Tool execution bonus (anti-exploitation: 3-gate design from ToolRL/ToolRLA).
    # Always emit the "tool_execution_bonus" key when the feature is enabled (0.0 when
    # no/failed tool use) so the per-sample reward_extra_info arrays stay rectangular.
    # "acc" is preserved as the raw task correctness so filter_groups.metric=acc keeps
    # filtering on correctness rather than the bonus-augmented score.
    if kwargs.get("tool_execution_bonus", False):
        tool_bonus = _tool_execution_bonus(solution_str, bonus=float(kwargs.get("tool_bonus_value", 0.2)))
        if not isinstance(score, dict):
            score = {"score": float(score), "acc": float(score)}
        else:
            score.setdefault("acc", score.get("score", 0.0))
        score["score"] = score.get("score", 0.0) + tool_bonus
        score["tool_execution_bonus"] = tool_bonus

    return score


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


def get_default_compute_score(reward_name: str | None):
    """Get the default compute_score function based on the reward manager type."""
    return default_compute_score


__all__ = ["default_compute_score"]
