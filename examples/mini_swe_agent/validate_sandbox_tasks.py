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
"""
Keep only the sandbox tasks whose grader behaves correctly on your sandbox backend.

For every row with ``extra_info.sandbox`` and ``extra_info.reference_solution``:

1. no-op: set up a fresh sandbox and run the eval command right away; it must FAIL;
2. oracle: set up another fresh sandbox, run the reference solution, then the eval command;
   it must PASS.

Run it with the same tool config as training (backend, image_root, timeouts), so tasks that
depend on the environment (kernel version, file ownership, directory order, ...) are dropped.
Writes the passing rows to <output_dir> under the same file names, plus report.jsonl.

    python3 examples/mini_swe_agent/validate_sandbox_tasks.py \\
        --data ~/data/openthoughts_agent_rl/train.parquet ~/data/openthoughts_agent_rl/test.parquet \\
        --tool_config examples/mini_swe_agent/config/bash_tool_config_singularity.yaml \\
        --output_dir ~/data/openthoughts_agent_rl_validated
"""

import argparse
import asyncio
import json
import os
import time
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from verl.tools.bash_tool import BashTool, CommandResult, SandboxSpec
from verl.tools.tool_registry import initialize_tools_from_config


async def _run_episode(tool: BashTool, spec: SandboxSpec, solution: str | None) -> CommandResult:
    session_id = uuid4().hex
    await tool.open_session(session_id, spec)
    try:
        if solution is not None:
            await tool.run_command(session_id, solution, timeout=tool.setup_timeout)
        timeout = spec.eval_timeout if spec.eval_timeout is not None else tool.eval_timeout
        return await tool.run_command(session_id, spec.eval_command, timeout=timeout)
    finally:
        await tool.close_session(session_id)


async def validate_row(tool: BashTool, extra_info: dict, semaphore: asyncio.Semaphore) -> dict:
    task_id = extra_info.get("task_id") or extra_info.get("index")
    async with semaphore:
        try:
            spec = SandboxSpec.from_value(extra_info["sandbox"])
            solution = extra_info.get("reference_solution")
            if not spec.eval_command or not solution:
                return {"task_id": task_id, "status": "error", "detail": "missing eval_command or reference_solution"}
            noop = await _run_episode(tool, spec, solution=None)
            oracle = await _run_episode(tool, spec, solution=solution)
        except Exception as e:
            return {"task_id": task_id, "status": "error", "detail": f"{type(e).__name__}: {e}"}
    if noop.returncode == 0:
        status = "noop_passes"
    elif oracle.returncode != 0:
        status = "oracle_fails"
    else:
        status = "ok"
    report = {"task_id": task_id, "status": status, "noop_returncode": noop.returncode}
    report["oracle_returncode"] = oracle.returncode
    if status != "ok":
        report["detail"] = (noop if status == "noop_passes" else oracle).output[-2000:]
    return report


async def main_async(args) -> None:
    tools = [t for t in initialize_tools_from_config(args.tool_config) if isinstance(t, BashTool)]
    if not tools:
        raise ValueError(f"No BashTool in {args.tool_config}")
    tool = tools[0]
    semaphore = asyncio.Semaphore(args.concurrency)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "report.jsonl"), "w") as report_file:
        for path in args.data:
            table = pq.read_table(os.path.expanduser(path))
            extra_infos = table.column("extra_info").to_pylist()
            start = time.monotonic()
            reports = await asyncio.gather(*(validate_row(tool, info, semaphore) for info in extra_infos))
            for report in reports:
                report_file.write(json.dumps({"file": os.path.basename(path), **report}) + "\n")

            keep = pa.array([report["status"] == "ok" for report in reports])
            out_path = os.path.join(args.output_dir, os.path.basename(path))
            pq.write_table(table.filter(keep), out_path)
            counts = {s: sum(r["status"] == s for r in reports) for s in ("ok", "noop_passes", "oracle_fails", "error")}
            print(f"{path}: {counts} in {time.monotonic() - start:.0f}s -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="Parquet files produced by a prepare_* script")
    parser.add_argument("--tool_config", required=True, help="Tool config with the BashTool used for training")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
