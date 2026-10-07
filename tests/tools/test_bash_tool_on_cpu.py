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
"""CPU tests for the sandboxed bash tool (local backend, docker argv, toy dataset)."""

import importlib.util
import os
import shutil
import subprocess
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verl.tools.bash_tool import BashTool, CommandResult, DockerEnvironment, SandboxSpec, _CappedBuffer

REPO_ROOT = Path(__file__).resolve().parents[2]


def _local_tool(**config) -> BashTool:
    return BashTool(config={"type": "native", "backend": "local", **config})


def _agent_data(session_id: str) -> SimpleNamespace:
    return SimpleNamespace(sandbox_session_id=session_id)


def test_default_schema():
    tool = _local_tool()
    assert tool.name == "bash"
    assert tool.tool_schema.function.parameters.required == ["command"]


def test_sandbox_spec_parsing():
    spec = SandboxSpec.from_value(
        '{"image": "python:3.11-slim", "cwd": "/w", "setup_commands": ["a", "b"], "eval_command": "c"}'
    )
    assert spec.image == "python:3.11-slim" and spec.cwd == "/w"
    assert spec.setup_commands == ["a", "b"] and spec.eval_command == "c"

    # Parquet round-trips: lists as numpy arrays, missing struct fields as None.
    spec = SandboxSpec.from_value({"setup_commands": np.array(["x"]), "env": None, "eval_timeout": 5})
    assert spec.setup_commands == ["x"] and spec.env == {} and spec.eval_timeout == 5.0

    assert SandboxSpec.from_value(None) == SandboxSpec()
    with pytest.raises(ValueError, match="Unknown sandbox spec keys"):
        SandboxSpec.from_value({"imag": "typo"})


@pytest.mark.asyncio
async def test_local_session_persists_files_but_not_shell_state():
    tool = _local_tool()
    await tool.open_session("s", SandboxSpec(setup_commands=["echo hello > greeting.txt"]))
    try:
        workdir = tool._sessions["s"].env.cwd
        response, _, metrics = await tool.execute("i", {"command": "cat greeting.txt"}, agent_data=_agent_data("s"))
        assert response.text == "<returncode>0</returncode>\n<output>\nhello\n</output>"
        assert metrics == {"returncode": 0, "timed_out": False}

        await tool.execute("i", {"command": "mkdir sub && cd sub && export FOO=1"}, agent_data=_agent_data("s"))
        response, _, _ = await tool.execute("i", {"command": 'pwd; echo "FOO=$FOO"'}, agent_data=_agent_data("s"))
        assert f"{workdir}\nFOO=\n" in response.text

        response, _, _ = await tool.execute("i", {"command": "echo oops >&2; exit 3"}, agent_data=_agent_data("s"))
        assert response.text.startswith("<returncode>3</returncode>") and "oops" in response.text
        assert tool.session_stats("s") == {"num_commands": 4, "num_command_timeouts": 0}
    finally:
        await tool.close_session("s")
    assert not os.path.exists(workdir)


@pytest.mark.asyncio
async def test_local_timeout_kills_process_group_and_keeps_partial_output():
    tool = _local_tool(command_timeout=1)
    await tool.open_session("s", SandboxSpec())
    try:
        start = time.monotonic()
        response, _, metrics = await tool.execute(
            "i", {"command": "echo started; sleep 30 & sleep 30"}, agent_data=_agent_data("s")
        )
        assert time.monotonic() - start < 10
        assert metrics["timed_out"] is True
        assert "timed out after 1s" in response.text and "started" in response.text
        assert tool.session_stats("s")["num_command_timeouts"] == 1
    finally:
        await tool.close_session("s")


@pytest.mark.asyncio
async def test_long_output_is_clipped_like_mini_swe_agent():
    tool = _local_tool(max_output_chars=100, max_capture_bytes=1000)
    await tool.open_session("s", SandboxSpec())
    try:
        response, _, _ = await tool.execute(
            "i", {"command": "seq 1 100000; echo LAST_LINE"}, agent_data=_agent_data("s")
        )
    finally:
        await tool.close_session("s")
    text = response.text
    assert text.startswith("<returncode>0</returncode>\n<warning>")
    assert "<output_head>\n1\n2\n" in text and "LAST_LINE" in text and "characters elided" in text
    assert len(text) < 1500


def test_capped_buffer_keeps_head_and_tail():
    buffer = _CappedBuffer(limit=10)
    for chunk in [b"abc", b"defghij", b"klmnop"]:
        buffer.feed(chunk)
    assert buffer.text() == "abcde\n... [6 bytes dropped] ...\nlmnop"

    small = _CappedBuffer(limit=10)
    small.feed(b"hi")
    assert small.text() == "hi"


@pytest.mark.asyncio
async def test_setup_failure_raises_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    tool = _local_tool()
    with pytest.raises(RuntimeError, match="exited with 7"):
        await tool.open_session("s", SandboxSpec(setup_commands=["touch made_it", "exit 7"]))
    assert "s" not in tool._sessions
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_execute_without_session_or_command():
    tool = _local_tool()
    response, _, metrics = await tool.execute("i", {"command": "ls"}, agent_data=_agent_data("missing"))
    assert "mini_swe_agent" in response.text and metrics["error"] == "no_session"

    await tool.open_session("s", SandboxSpec())
    try:
        response, _, _ = await tool.execute("i", {"command": "  "}, agent_data=_agent_data("s"))
        assert "non-empty" in response.text
        assert tool.session_stats("s")["num_commands"] == 0
    finally:
        await tool.close_session("s")


def test_docker_argv():
    env = DockerEnvironment(
        image="python:3.11-slim",
        cwd="/workspace",
        env={"PAGER": "cat"},
        shell=["bash", "-lc"],
        executable="podman",
        run_args=["--rm", "--network=none"],
        container_lifetime="2h",
        start_timeout=10,
        max_capture_bytes=1000,
    )
    argv = env.run_argv()
    assert argv[:5] == ["podman", "run", "-d", "--name", argv[4]] and argv[4].startswith("verl-bash-")
    assert argv[5:] == ["-w", "/workspace", "--rm", "--network=none", "python:3.11-slim", "sleep", "2h"]
    env.container_id = "abc123"
    assert env.exec_argv("ls -la") == [
        "podman", "exec", "-w", "/workspace", "-e", "PAGER=cat", "abc123", "bash", "-lc", "ls -la"
    ]  # fmt: skip


def test_docker_backend_requires_image():
    tool = BashTool(config={"type": "native", "backend": "docker"})
    with pytest.raises(ValueError, match="needs an image"):
        tool._make_environment(SandboxSpec())


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _docker_available(), reason="docker daemon not reachable")
@pytest.mark.asyncio
async def test_docker_session_end_to_end():
    image = os.environ.get("VERL_TEST_SANDBOX_IMAGE", "python:3.11-slim")
    tool = BashTool(config={"type": "native", "backend": "docker", "default_cwd": "/workspace"})
    await tool.open_session("s", SandboxSpec(image=image, setup_commands=["echo 1 > f.txt"]))
    container_id = tool._sessions["s"].env.container_id
    try:
        response, _, _ = await tool.execute("i", {"command": "pwd && cat f.txt"}, agent_data=_agent_data("s"))
        assert "/workspace\n1\n" in response.text
    finally:
        await tool.close_session("s")
    inspect = subprocess.run(["docker", "inspect", container_id], capture_output=True)
    assert inspect.returncode != 0


def test_singularity_argv_and_image_resolution(tmp_path):
    tool = BashTool(
        config={
            "type": "native",
            "backend": "singularity",
            "singularity_executable": "apptainer",
            "singularity_exec_args": ["--no-mount", "cwd"],
            "image_root": str(tmp_path),
            "env": {"FOO": "1"},
        }
    )
    assert tool.resolve_image("python:3.11-slim") == f"{tmp_path}/python_3.11-slim.sif"
    assert tool.resolve_image("ghcr.io/org/img:tag") == f"{tmp_path}/ghcr.io_org_img_tag.sif"
    assert tool.resolve_image("/abs/x.sif") == "/abs/x.sif"
    assert tool.resolve_image("docker://python:3.11-slim") == "docker://python:3.11-slim"

    env = tool._make_environment(SandboxSpec(image="python:3.11-slim"))
    env.workdir = "/host/work"
    argv = env.exec_argv("ls")
    assert argv[:5] == ["apptainer", "--silent", "exec", "--containall", "--cleanenv"]
    assert argv[5:9] == ["--pwd", "/workspace", "--bind", "/host/work:/workspace"]
    assert "--env" in argv and "FOO=1" in argv and "PAGER=cat" in argv
    assert argv[-6:] == ["--no-mount", "cwd", f"{tmp_path}/python_3.11-slim.sif", "bash", "-lc", "ls"]

    with pytest.raises(ValueError, match="other than '/'"):
        tool._make_environment(SandboxSpec(image="x.sif", cwd="/"))


def _singularity_executable() -> str | None:
    candidates = [os.environ.get("VERL_TEST_SINGULARITY_EXECUTABLE"), "singularity", "apptainer"]
    return next((c for c in candidates if c and shutil.which(c)), None)


SINGULARITY_IMAGE = os.environ.get("VERL_TEST_SINGULARITY_IMAGE", "")
requires_singularity = pytest.mark.skipif(
    _singularity_executable() is None or not os.path.isfile(SINGULARITY_IMAGE),
    reason="needs singularity/apptainer and VERL_TEST_SINGULARITY_IMAGE pointing to a python .sif",
)


def _singularity_tool(**config) -> BashTool:
    return BashTool(
        config={
            "type": "native",
            "backend": "singularity",
            "singularity_executable": _singularity_executable(),
            "default_image": SINGULARITY_IMAGE,
            **config,
        }
    )


@requires_singularity
@pytest.mark.asyncio
async def test_singularity_seeds_cwd_and_isolates(monkeypatch):
    monkeypatch.setenv("VERL_TEST_SECRET", "leak")
    tool = _singularity_tool(env={"FOO": "bar"}, command_timeout=2)
    # /usr/local/bin exists in python images: its contents are copied into the writable cwd.
    await tool.open_session("s", SandboxSpec(cwd="/usr/local/bin"))
    workdir = tool._sessions["s"].env.workdir
    try:
        result = await tool.run_command("s", "ls; touch created; mkdir ro && chmod 555 ro")
        assert result.returncode == 0 and "python3" in result.output.split()
        result = await tool.run_command("s", 'pwd; ls created; echo "FOO=$FOO SECRET=$VERL_TEST_SECRET"')
        assert result.output == "/usr/local/bin\ncreated\nFOO=bar SECRET=\n"
        result = await tool.run_command("s", "touch /usr/lib/x")
        assert result.returncode != 0 and "Read-only" in result.output

        start = time.monotonic()
        result = await tool.run_command("s", "echo started; sleep 30")
        assert result.timed_out and "started" in result.output and time.monotonic() - start < 15
    finally:
        await tool.close_session("s")
    assert not os.path.exists(workdir)

    # A new session starts from the pristine image again.
    await tool.open_session("s2", SandboxSpec(cwd="/usr/local/bin"))
    try:
        result = await tool.run_command("s2", "ls created")
        assert result.returncode != 0
    finally:
        await tool.close_session("s2")


@requires_singularity
@pytest.mark.asyncio
async def test_singularity_runs_toy_task():
    toy = _load_toy_data_module()
    sandbox, _ = toy.make_task("sum_to", "dedupe", image="python:3.11-slim", cwd="/workspace")
    sandbox["image"] = SINGULARITY_IMAGE
    tool = _singularity_tool()
    await tool.open_session("s", SandboxSpec.from_value(sandbox))
    try:
        before = await tool.run_command("s", sandbox["eval_command"])
        await tool.run_command("s", "sed -i 's/range(1, n)/range(1, n + 1)/' *.py")
        after = await tool.run_command("s", sandbox["eval_command"])
    finally:
        await tool.close_session("s")
    assert before.returncode != 0 and "FAILED" in before.output
    assert after.returncode == 0, after.output


def _load_toy_data_module():
    path = REPO_ROOT / "examples" / "mini_swe_agent" / "prepare_toy_bugfix_data.py"
    spec = importlib.util.spec_from_file_location("prepare_toy_bugfix_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tests_pass(code: str, fn_name: str, asserts: list[str]) -> bool:
    namespace: dict = {}
    exec(textwrap.dedent(code), namespace)
    case = unittest.TestCase()
    try:
        for line in asserts:
            exec(line, {"self": case, fn_name: namespace[fn_name]})
    except AssertionError:
        return False
    return True


def test_toy_tasks_fail_when_buggy_and_pass_when_fixed():
    toy = _load_toy_data_module()
    toy.random.seed(0)
    # Templates draw random values, so check many draws: a draw where the buggy code
    # still passes would be an unsolvable task.
    for name, (template, fn_names) in toy.TEMPLATES.items():
        for _ in range(50):
            buggy, fixed, asserts = template(fn_names[0])
            assert not _tests_pass(buggy, fn_names[0], asserts), f"{name}: buggy version passes {asserts}"
            assert _tests_pass(fixed, fn_names[0], asserts), f"{name}: fixed version fails {asserts}"


@pytest.mark.asyncio
async def test_toy_task_setup_and_eval_commands_run():
    toy = _load_toy_data_module()
    sandbox, task = toy.make_task("sum_to", "dedupe", image="python:3.11-slim", cwd="/workspace")
    assert "Do not modify" in task
    tool = _local_tool()
    await tool.open_session("s", SandboxSpec.from_value(sandbox))
    try:
        before = await tool.run_command("s", sandbox["eval_command"])
        await tool.run_command("s", "sed -i 's/range(1, n)/range(1, n + 1)/' *.py")
        after: CommandResult = await tool.run_command("s", sandbox["eval_command"])
    finally:
        await tool.close_session("s")
    assert before.returncode != 0 and "FAILED" in before.output
    assert after.returncode == 0, after.output
