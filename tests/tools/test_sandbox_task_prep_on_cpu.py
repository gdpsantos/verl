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
"""CPU tests for the OpenThoughts-Agent-v1-RL converter and the sandbox task validator."""

import asyncio
import base64
import importlib.util
import io
import os
import tarfile
from pathlib import Path

import pytest

from verl.tools.bash_tool import BashTool, SandboxSpec

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "mini_swe_agent"
SINGULARITY_IMAGE = os.environ.get("VERL_TEST_SINGULARITY_IMAGE", "")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task_archive(files: dict[str, bytes], mode: int = 0o644) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.uid = len(data), mode, 505
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


SOLVE = (
    "#!/usr/bin/env bash\nset -euo pipefail\ncd /workspace\n"
    "bash -lc 'cat sub/b.txt a.txt' > command_capture.txt 2>&1 || true\n"
    'cp command_capture.txt "/output/command_capture.txt"\n'
)
TASK = {
    "instruction.md": b"# Task\nConcatenate sub/b.txt and a.txt into /output/command_capture.txt.\n",
    "tests/expected_output.txt": b"B\nA\n",
    "solution/solve.sh": SOLVE.encode(),
    "environment/seeds/a.txt": b"A\n",
    "environment/seeds/sub/b.txt": b"B\n",
    "environment/seeds/blob.bin": bytes(range(256)),
}


def test_convert_task_builds_sandbox_row():
    prep = _load("prepare_openthoughts_agent_rl")
    row = prep.convert_task("task_1", _task_archive(TASK), image="ot.sif", max_inline_bytes=100_000)

    assert row["agent_name"] == "mini_swe_agent" and row["data_source"] == "sandbox_eval/openthoughts_agent_rl"
    assert row["prompt"][1]["content"].startswith("# Task")
    info = row["extra_info"]
    assert info["task_id"] == "task_1" and info["reference_solution"] == SOLVE
    spec = SandboxSpec.from_value(info["sandbox"])
    assert spec.image == "ot.sif" and spec.cwd == "/workspace" and spec.writable_paths == ["/output"]
    assert spec.env["TZ"] == "UTC"

    # The seed tarball unpacks to the original files, owned by root as in the Dockerfile's COPY.
    payload = spec.setup_commands[1].split("printf %s ")[1].split(" |")[0]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz") as tf:
        members = {m.name: m for m in tf.getmembers()}
        assert tf.extractfile(members["./sub/b.txt"]).read() == b"B\n"
        assert tf.extractfile(members["./blob.bin"]).read() == bytes(range(256))
        assert {m.uname for m in members.values()} == {"root"}
    # The expected output only appears inside the eval command, never in the prompt.
    assert base64.b64encode(b"B\nA\n").decode() in spec.eval_command
    assert "B\nA" not in row["prompt"][1]["content"]


def test_convert_task_skips_seeds_too_large_to_inline():
    prep = _load("prepare_openthoughts_agent_rl")
    task = {**TASK, "environment/seeds/random.bin": os.urandom(200_000)}
    assert prep.convert_task("big", _task_archive(task), image="ot.sif", max_inline_bytes=100_000) is None


def _local_row(setup: str, eval_command: str, solution: str) -> dict:
    return {
        "task_id": "t",
        "sandbox": {"setup_commands": [setup], "eval_command": eval_command},
        "reference_solution": solution,
    }


@pytest.mark.parametrize(
    "solution, eval_command, status",
    [
        ("echo done > out.txt", "grep -q done out.txt", "ok"),
        ("echo done > out.txt", "true", "noop_passes"),
        ("echo wrong > out.txt", "grep -q done out.txt", "oracle_fails"),
    ],
)
def test_validate_row_statuses(solution, eval_command, status):
    validate = _load("validate_sandbox_tasks")
    tool = BashTool(config={"type": "native", "backend": "local"})
    row = _local_row("echo start > in.txt", eval_command, solution)
    report = asyncio.run(validate.validate_row(tool, row, asyncio.Semaphore(1)))
    assert report["status"] == status
    assert tool._sessions == {}


def test_validate_row_reports_setup_errors():
    validate = _load("validate_sandbox_tasks")
    tool = BashTool(config={"type": "native", "backend": "local"})
    report = asyncio.run(validate.validate_row(tool, _local_row("exit 3", "true", "true"), asyncio.Semaphore(1)))
    assert report["status"] == "error" and "exited with 3" in report["detail"]


def _singularity_executable():
    import shutil

    candidates = [os.environ.get("VERL_TEST_SINGULARITY_EXECUTABLE"), "singularity", "apptainer"]
    return next((c for c in candidates if c and shutil.which(c)), None)


@pytest.mark.skipif(
    _singularity_executable() is None or not os.path.isfile(SINGULARITY_IMAGE),
    reason="needs singularity/apptainer and VERL_TEST_SINGULARITY_IMAGE",
)
def test_converted_task_validates_in_singularity():
    prep, validate = _load("prepare_openthoughts_agent_rl"), _load("validate_sandbox_tasks")
    tool = BashTool(
        config={"type": "native", "backend": "singularity", "singularity_executable": _singularity_executable()}
    )
    row = prep.convert_task("task_1", _task_archive(TASK), image=SINGULARITY_IMAGE, max_inline_bytes=100_000)
    report = asyncio.run(validate.validate_row(tool, row["extra_info"], asyncio.Semaphore(1)))
    assert report["status"] == "ok", report

    broken = {**row["extra_info"], "reference_solution": SOLVE.replace("sub/b.txt a.txt", "a.txt sub/b.txt")}
    report = asyncio.run(validate.validate_row(tool, broken, asyncio.Semaphore(1)))
    assert report["status"] == "oracle_fails" and "Output mismatch" in report["detail"]
