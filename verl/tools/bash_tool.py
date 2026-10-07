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
"""Bash tool backed by a per-trajectory sandbox, in the style of mini-swe-agent.

Every command runs in a fresh shell inside the trajectory's sandbox, so the file
system persists across turns while ``cd`` and environment variables do not.
Sandbox sessions are opened and closed by
:class:`verl.experimental.agent_loop.mini_swe_agent_loop.MiniSweAgentLoop`; the
tool only executes commands in the session attached to the trajectory.
"""

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import signal
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Non-interactive defaults so pagers and progress bars do not stall or flood the output.
DEFAULT_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

DEFAULT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in the task's sandbox and return its exit code and output. "
            "Each call runs in a new shell: files persist between calls, but directory changes "
            "and environment variables do not (prefix commands with `cd /path &&` when needed). "
            "Interactive programs (editors, pagers) are not supported."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute."},
            },
            "required": ["command"],
        },
    },
}


@dataclass
class SandboxSpec:
    """Per-sample sandbox description, read from ``extra_info["sandbox"]``.

    Attributes:
        image: Docker image, or for the singularity backend a .sif path or an image name
            resolved under the tool's ``image_root``. Ignored by the local backend.
        cwd: Working directory inside the container. With singularity it is the only writable,
            persistent path. Ignored by the local backend, which uses a fresh temporary directory.
        env: Extra environment variables for every command.
        setup_commands: Commands run before the episode; any non-zero exit aborts the rollout.
        eval_command: Command run after the episode; exit code 0 means the task is solved.
        eval_timeout: Optional per-sample override of the tool's ``eval_timeout``.
    """

    image: Optional[str] = None
    cwd: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    setup_commands: list[str] = field(default_factory=list)
    eval_command: Optional[str] = None
    eval_timeout: Optional[float] = None

    @classmethod
    def from_value(cls, value: Any) -> "SandboxSpec":
        """Build a spec from a dict or a JSON string (as stored in parquet)."""
        if value is None:
            return cls()
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise TypeError(f"Sandbox spec must be a dict or JSON string, got {type(value).__name__}")
        unknown = set(value) - {"image", "cwd", "env", "setup_commands", "eval_command", "eval_timeout"}
        if unknown:
            raise ValueError(f"Unknown sandbox spec keys: {sorted(unknown)}")
        # Parquet round-trips may turn lists into numpy arrays and missing struct fields into None.
        setup_commands = value.get("setup_commands")
        if isinstance(setup_commands, str):
            setup_commands = [setup_commands]
        eval_timeout = value.get("eval_timeout")
        return cls(
            image=value.get("image") or None,
            cwd=value.get("cwd") or None,
            env={str(k): str(v) for k, v in (value.get("env") or {}).items() if v is not None},
            setup_commands=[str(c) for c in setup_commands] if setup_commands is not None else [],
            eval_command=value.get("eval_command") or None,
            eval_timeout=float(eval_timeout) if eval_timeout is not None else None,
        )


@dataclass
class CommandResult:
    returncode: Optional[int]
    output: str
    timed_out: bool = False


class _CappedBuffer:
    """Keeps the head and tail of a byte stream, dropping the middle beyond ``limit`` bytes."""

    def __init__(self, limit: int):
        self.half = max(limit // 2, 1)
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if len(self.head) < self.half:
            take = self.half - len(self.head)
            self.head += chunk[:take]
            chunk = chunk[take:]
        if chunk:
            self.tail += chunk
            if len(self.tail) > self.half:
                del self.tail[: len(self.tail) - self.half]

    def text(self) -> str:
        dropped = self.total - len(self.head) - len(self.tail)
        middle = f"\n... [{dropped} bytes dropped] ...\n".encode() if dropped > 0 else b""
        return (bytes(self.head) + middle + bytes(self.tail)).decode("utf-8", errors="replace")


async def _run_process(
    argv: Sequence[str],
    timeout: float,
    max_capture_bytes: int,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    kill_process_group: bool = False,
) -> CommandResult:
    """Run ``argv`` with stdout and stderr merged, keeping partial output on timeout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    buffer = _CappedBuffer(max_capture_bytes)

    async def _drain():
        while chunk := await proc.stdout.read(65536):
            buffer.feed(chunk)
        return await proc.wait()

    try:
        returncode = await asyncio.wait_for(_drain(), timeout=timeout)
        return CommandResult(returncode=returncode, output=buffer.text())
    except asyncio.TimeoutError:
        try:
            if kill_process_group:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return CommandResult(returncode=None, output=buffer.text(), timed_out=True)


class SandboxEnvironment(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def execute(self, command: str, timeout: float) -> CommandResult: ...

    @abstractmethod
    async def stop(self) -> None: ...


class LocalEnvironment(SandboxEnvironment):
    """Runs commands on the host in a fresh temporary directory.

    There is no isolation: the model's commands run with the trainer's permissions.
    Use it only for debugging and tests.
    """

    def __init__(self, shell: list[str], env: dict[str, str], max_capture_bytes: int):
        self.shell = shell
        self.env = env
        self.max_capture_bytes = max_capture_bytes
        self.cwd: Optional[str] = None

    async def start(self) -> None:
        self.cwd = tempfile.mkdtemp(prefix="verl-bash-")

    async def execute(self, command: str, timeout: float) -> CommandResult:
        return await _run_process(
            [*self.shell, command],
            timeout=timeout,
            max_capture_bytes=self.max_capture_bytes,
            cwd=self.cwd,
            env={**os.environ, **self.env},
            kill_process_group=True,
        )

    async def stop(self) -> None:
        if self.cwd:
            await asyncio.to_thread(_force_rmtree, self.cwd)
            self.cwd = None


def _force_rmtree(path: str) -> None:
    """rmtree that also removes trees containing read-only directories (e.g. copied from an image)."""
    for root, dirs, _ in os.walk(path):
        for name in dirs:
            try:
                os.chmod(os.path.join(root, name), 0o700)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


class SingularityEnvironment(SandboxEnvironment):
    """Runs every command with its own ``singularity exec`` of a SIF image.

    Only ``cwd`` is writable and persists across commands: it is a host directory bound
    into the container, seeded at start with whatever the image has at that path (e.g. a
    repository). Everything else is the read-only image, with a private /tmp and home
    (``--containall``) and none of the trainer's environment variables (``--cleanenv``).
    No process outlives a command, so nothing is left behind if a worker crashes.
    Works with SingularityCE 3.x/4.x and Apptainer (``executable: apptainer``).
    """

    SEED_MOUNT = "/verl_seed"

    def __init__(
        self,
        image: str,
        cwd: str,
        env: dict[str, str],
        shell: list[str],
        executable: str,
        exec_args: list[str],
        work_root: Optional[str],
        start_timeout: float,
        max_capture_bytes: int,
    ):
        if cwd == "/":
            raise ValueError("The singularity backend needs a cwd other than '/' (it is bound to a host directory)")
        self.image = image
        self.cwd = cwd
        self.env = env
        self.shell = shell
        self.executable = executable
        self.exec_args = exec_args
        self.work_root = work_root
        self.start_timeout = start_timeout
        self.max_capture_bytes = max_capture_bytes
        self.workdir: Optional[str] = None

    def _base_argv(self) -> list[str]:
        return [self.executable, "--silent", "exec", "--containall", "--cleanenv"]

    def exec_argv(self, command: str) -> list[str]:
        env_args = [arg for key, value in self.env.items() for arg in ("--env", f"{key}={value}")]
        return [
            *self._base_argv(),
            "--pwd",
            self.cwd,
            "--bind",
            f"{self.workdir}:{self.cwd}",
            *env_args,
            *self.exec_args,
            self.image,
            *self.shell,
            command,
        ]

    def seed_argv(self) -> list[str]:
        cwd = shlex.quote(self.cwd)
        script = (
            f"if [ -d {cwd} ]; then cd {cwd} && "
            f"if command -v tar >/dev/null 2>&1; then tar cf - . | tar xf - -C {self.SEED_MOUNT}; "
            f"else cp -R . {self.SEED_MOUNT}/; fi; fi"
        )
        return [*self._base_argv(), "--bind", f"{self.workdir}:{self.SEED_MOUNT}", self.image, "sh", "-c", script]

    async def start(self) -> None:
        if "://" not in self.image and not os.path.exists(self.image):
            raise FileNotFoundError(f"Singularity image not found: {self.image}")
        self.workdir = tempfile.mkdtemp(prefix="verl-bash-", dir=self.work_root)
        result = await _run_process(
            self.seed_argv(), self.start_timeout, self.max_capture_bytes, kill_process_group=True
        )
        if result.timed_out or result.returncode != 0:
            await self.stop()
            raise RuntimeError(f"Failed to start sandbox from image {self.image!r}: {result.output.strip()}")

    async def execute(self, command: str, timeout: float) -> CommandResult:
        return await _run_process(self.exec_argv(command), timeout, self.max_capture_bytes, kill_process_group=True)

    async def stop(self) -> None:
        if self.workdir:
            await asyncio.to_thread(_force_rmtree, self.workdir)
            self.workdir = None


class DockerEnvironment(SandboxEnvironment):
    """Runs every command with ``docker exec`` in a long-lived container.

    The container runs ``sleep <container_lifetime>`` so that a container orphaned by a
    crashed worker exits on its own; with ``--rm`` in ``run_args`` it is also removed.
    """

    def __init__(
        self,
        image: str,
        cwd: str,
        env: dict[str, str],
        shell: list[str],
        executable: str,
        run_args: list[str],
        container_lifetime: str,
        start_timeout: float,
        max_capture_bytes: int,
    ):
        self.image = image
        self.cwd = cwd
        self.env = env
        self.shell = shell
        self.executable = executable
        self.run_args = run_args
        self.container_lifetime = container_lifetime
        self.start_timeout = start_timeout
        self.max_capture_bytes = max_capture_bytes
        self.container_id: Optional[str] = None

    def _env_args(self) -> list[str]:
        return [arg for key, value in self.env.items() for arg in ("-e", f"{key}={value}")]

    def run_argv(self) -> list[str]:
        name = f"verl-bash-{uuid4().hex[:12]}"
        return [
            self.executable,
            "run",
            "-d",
            "--name",
            name,
            "-w",
            self.cwd,
            *self.run_args,
            self.image,
            "sleep",
            self.container_lifetime,
        ]

    def exec_argv(self, command: str) -> list[str]:
        return [self.executable, "exec", "-w", self.cwd, *self._env_args(), self.container_id, *self.shell, command]

    async def start(self) -> None:
        result = await _run_process(self.run_argv(), self.start_timeout, self.max_capture_bytes)
        if result.timed_out or result.returncode != 0:
            raise RuntimeError(f"Failed to start container from image {self.image!r}: {result.output.strip()}")
        self.container_id = result.output.strip().splitlines()[-1]

    async def execute(self, command: str, timeout: float) -> CommandResult:
        # On timeout only the docker client is killed; a runaway process keeps running in
        # the container until the container is removed at the end of the episode.
        return await _run_process(self.exec_argv(command), timeout, self.max_capture_bytes)

    async def stop(self) -> None:
        if self.container_id is None:
            return
        result = await _run_process(
            [self.executable, "rm", "-f", self.container_id], self.start_timeout, self.max_capture_bytes
        )
        if result.returncode != 0:
            logger.warning(f"[BashTool] Failed to remove container {self.container_id}: {result.output.strip()}")
        self.container_id = None


@dataclass
class _Session:
    env: SandboxEnvironment
    num_commands: int = 0
    num_timeouts: int = 0


class BashTool(BaseTool):
    """Bash tool executing commands in the trajectory's sandbox session.

    Config keys (all optional):
        backend: "docker" (default), "singularity" (also Apptainer; for clusters without
            Docker) or "local" (no isolation, for debugging only).
        command_timeout: Seconds before a model command is killed (default 60).
        setup_timeout: Seconds allowed for each setup command (default 600).
        eval_timeout: Seconds allowed for the evaluation command (default 600).
        max_output_chars: Output longer than this is shown as head and tail with a warning,
            as in mini-swe-agent (default 10000).
        max_capture_bytes: Hard cap on bytes kept from a command's output (default 1000000).
        env: Environment variables added to every command (on top of pager/progress defaults).
        shell: Command prefix, default ["bash", "-lc"] for containers and ["bash", "-c"] for local.
        default_image / default_cwd: Used when the sample's sandbox spec omits them. default_cwd
            defaults to "/" for docker and "/workspace" for singularity.
        singularity_executable: "singularity" (default) or "apptainer"; it must be on PATH of the
            Ray workers (e.g. ``module load singularity`` before ``ray start``) or an absolute path.
        singularity_exec_args: Extra ``singularity exec`` arguments (default []).
        image_root: Directory of .sif files. A relative image name such as "python:3.11-slim"
            resolves to "<image_root>/python_3.11-slim.sif"; absolute paths and URIs are used as is.
        work_root: Where per-trajectory work directories are created (default: $TMPDIR).
        docker_executable: Container CLI, e.g. "docker" (default) or "podman".
        docker_run_args: Extra ``docker run`` arguments (default ["--rm"]).
        container_lifetime: Argument to ``sleep`` in the container (default "4h").
        start_timeout: Seconds allowed for starting/removing a container (default 300).
        max_concurrent_starts: Limit on sandboxes starting at once per worker (default 16).
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        super().__init__(config, tool_schema)
        self.backend = config.get("backend", "docker")
        if self.backend not in ("docker", "singularity", "local"):
            raise ValueError(f"Unknown BashTool backend {self.backend!r}, expected 'docker', 'singularity' or 'local'")
        self.command_timeout = float(config.get("command_timeout", 60))
        self.setup_timeout = float(config.get("setup_timeout", 600))
        self.eval_timeout = float(config.get("eval_timeout", 600))
        self.max_output_chars = int(config.get("max_output_chars", 10000))
        self.max_capture_bytes = int(config.get("max_capture_bytes", 1_000_000))
        self.env = {**DEFAULT_ENV, **{str(k): str(v) for k, v in (config.get("env") or {}).items()}}
        default_shell = ["bash", "-c"] if self.backend == "local" else ["bash", "-lc"]
        self.shell = list(config.get("shell") or default_shell)
        self.default_image = config.get("default_image")
        self.default_cwd = config.get("default_cwd", "/workspace" if self.backend == "singularity" else "/")
        self.singularity_executable = config.get("singularity_executable", "singularity")
        self.singularity_exec_args = list(config.get("singularity_exec_args") or [])
        self.image_root = config.get("image_root")
        self.work_root = config.get("work_root")
        self.docker_executable = config.get("docker_executable", "docker")
        self.docker_run_args = list(config.get("docker_run_args", ["--rm"]))
        self.container_lifetime = str(config.get("container_lifetime", "4h"))
        self.start_timeout = float(config.get("start_timeout", 300))
        self.max_concurrent_starts = int(config.get("max_concurrent_starts", 16))

        self._sessions: dict[str, _Session] = {}
        self._start_semaphore: Optional[asyncio.Semaphore] = None
        logger.info(f"Initialized BashTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        if getattr(self, "tool_schema", None) is None:
            return OpenAIFunctionToolSchema.model_validate(DEFAULT_TOOL_SCHEMA)
        return self.tool_schema

    def resolve_image(self, image: str) -> str:
        """Map a Docker-style image name to a .sif file under ``image_root`` (singularity backend)."""
        if "://" in image or os.path.isabs(image) or not self.image_root:
            return image
        name = image if image.endswith(".sif") else re.sub(r"[/:@]", "_", image) + ".sif"
        return os.path.join(self.image_root, name)

    def _make_environment(self, spec: SandboxSpec) -> SandboxEnvironment:
        env = {**self.env, **spec.env}
        if self.backend == "local":
            return LocalEnvironment(shell=self.shell, env=env, max_capture_bytes=self.max_capture_bytes)
        image = spec.image or self.default_image
        if not image:
            raise ValueError(
                f"The {self.backend} backend needs an image: set extra_info.sandbox.image or default_image"
            )
        if self.backend == "singularity":
            return SingularityEnvironment(
                image=self.resolve_image(image),
                cwd=spec.cwd or self.default_cwd,
                env=env,
                shell=self.shell,
                executable=self.singularity_executable,
                exec_args=self.singularity_exec_args,
                work_root=self.work_root,
                start_timeout=self.start_timeout,
                max_capture_bytes=self.max_capture_bytes,
            )
        return DockerEnvironment(
            image=image,
            cwd=spec.cwd or self.default_cwd,
            env=env,
            shell=self.shell,
            executable=self.docker_executable,
            run_args=self.docker_run_args,
            container_lifetime=self.container_lifetime,
            start_timeout=self.start_timeout,
            max_capture_bytes=self.max_capture_bytes,
        )

    async def open_session(self, session_id: str, spec: SandboxSpec) -> None:
        """Start a sandbox for one trajectory and run the spec's setup commands."""
        if session_id in self._sessions:
            raise ValueError(f"Sandbox session {session_id} already exists")
        if self._start_semaphore is None:
            self._start_semaphore = asyncio.Semaphore(self.max_concurrent_starts)
        environment = self._make_environment(spec)
        try:
            async with self._start_semaphore:
                await environment.start()
            for command in spec.setup_commands:
                result = await environment.execute(command, self.setup_timeout)
                if result.timed_out or result.returncode != 0:
                    status = "timed out" if result.timed_out else f"exited with {result.returncode}"
                    raise RuntimeError(f"Sandbox setup command {status}: {command[:200]!r}\n{result.output[-2000:]}")
        except BaseException:
            await environment.stop()
            raise
        self._sessions[session_id] = _Session(env=environment)

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.env.stop()

    async def run_command(self, session_id: str, command: str, timeout: Optional[float] = None) -> CommandResult:
        """Run a command in an open session (used for both model commands and evaluation)."""
        session = self._sessions[session_id]
        return await session.env.execute(command, timeout if timeout is not None else self.command_timeout)

    def session_stats(self, session_id: str) -> dict[str, int]:
        session = self._sessions[session_id]
        return {"num_commands": session.num_commands, "num_command_timeouts": session.num_timeouts}

    def format_observation(self, command: str, result: CommandResult) -> str:
        """Render a command result the way mini-swe-agent shows it to the model."""
        if result.timed_out:
            return (
                f"The last command <command>{command}</command> timed out after {self.command_timeout:g}s "
                "and has been killed.\n"
                f"The output of the command was:\n<output>\n{self._clip(result.output)}\n</output>\n"
                "Please try another command and make sure to avoid those requiring interactive input."
            )
        output = result.output
        if len(output) <= self.max_output_chars:
            return f"<returncode>{result.returncode}</returncode>\n<output>\n{output}</output>"
        return (
            f"<returncode>{result.returncode}</returncode>\n"
            "<warning>\nThe output of your last command was too long.\n"
            "Please try a different command that does not produce as much output.\n"
            "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines "
            "selectively.\nIf you're using grep or find and it produced too much output, you can use a more "
            "selective search pattern.\n</warning>\n" + self._clip(output)
        )

    def _clip(self, output: str) -> str:
        if len(output) <= self.max_output_chars:
            return output
        half = self.max_output_chars // 2
        elided = len(output) - 2 * half
        return (
            f"<output_head>\n{output[:half]}\n</output_head>\n"
            f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
            f"<output_tail>\n{output[-half:]}\n</output_tail>"
        )

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        session_id = getattr(agent_data, "sandbox_session_id", None)
        if session_id is None or session_id not in self._sessions:
            msg = "Error: no sandbox session for this trajectory. The bash tool requires the mini_swe_agent agent loop."
            logger.error(f"[BashTool] {msg}")
            return ToolResponse(text=msg), 0.0, {"error": "no_session"}

        command = parameters.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResponse(text="Error: 'command' must be a non-empty string."), 0.0, {"error": "bad_command"}

        session = self._sessions[session_id]
        session.num_commands += 1
        try:
            result = await session.env.execute(command, self.command_timeout)
        except Exception as e:
            logger.error(f"[BashTool] Command execution failed: {e}")
            return ToolResponse(text=f"Error: command execution failed: {e}"), 0.0, {"error": str(e)}
        if result.timed_out:
            session.num_timeouts += 1
        metrics = {"returncode": result.returncode, "timed_out": result.timed_out}
        return ToolResponse(text=self.format_observation(command, result)), 0.0, metrics
