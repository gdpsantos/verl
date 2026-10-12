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
Convert OpenThoughts-Agent-v1-RL (728 nl2bash terminal tasks) for the mini_swe_agent loop.

Every task shares one Ubuntu image (see images/openthoughts_agent_rl.def), so a task only
needs its seed files and its grader:

- setup: unpack the task's seed files into /workspace (inlined as a base64 tar.gz);
- eval: /output/command_capture.txt must exist, be non-empty and equal the expected output,
  as in the dataset's tests/test.sh;
- the reference solution (solution/solve.sh) is kept in extra_info for validate_sandbox_tasks.py.

Download tasks.parquet where there is internet access:
    https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL/resolve/main/tasks.parquet
"""

import argparse
import base64
import copy
import io
import os
import random
import tarfile

import datasets
import pyarrow.parquet as pq

DATA_SOURCE = "sandbox_eval/openthoughts_agent_rl"
WORKSPACE = "/workspace"
OUTPUT_DIR = "/output"
CAPTURE = f"{OUTPUT_DIR}/command_capture.txt"
SANDBOX_ENV = {"TZ": "UTC", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
# Linux limits a single command-line argument to 128 KiB; stay well below it.
DEFAULT_MAX_INLINE_BYTES = 100_000

SYSTEM_PROMPT = (
    "You are an expert Linux terminal user working in a sandboxed Ubuntu machine. Use the `bash` tool "
    "to inspect files and run commands. Each tool call runs in a new shell: files persist between calls, "
    "but `cd` and environment variables do not, so prefix commands with `cd <dir> &&` when needed. "
    "Interactive programs are not available. Run one command per tool call and check the results. "
    "When the task is complete, reply with a short summary and do not call the tool again."
)


def _b64_tar(members: list[tuple[tarfile.TarInfo, bytes]]) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        for info, data in members:
            info = copy.copy(info)
            info.uid, info.gid, info.uname, info.gname = 0, 0, "root", "root"
            tf.addfile(info, io.BytesIO(data) if info.isfile() else None)
    return base64.b64encode(buf.getvalue()).decode()


def _eval_command(expected: bytes) -> str:
    encoded = base64.b64encode(expected).decode()
    return (
        f'expected="$(mktemp)" && printf %s {encoded} | base64 -d > "$expected"\n'
        f'if [ ! -s {CAPTURE} ]; then echo "Missing or empty {CAPTURE}"; exit 1; fi\n'
        f'if cmp -s "$expected" {CAPTURE}; then echo "Output matches"; exit 0; fi\n'
        f'echo "Output mismatch"; echo "--- expected"; head -c 2000 "$expected"; '
        f'echo "--- actual"; head -c 2000 {CAPTURE}; exit 1'
    )


def convert_task(name: str, archive: bytes, image: str, max_inline_bytes: int) -> dict | None:
    """Build one dataset row, or return None if the seeds are too large to inline."""
    files: dict[str, bytes] = {}
    seeds: list[tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tf:
        for member in tf.getmembers():
            if not (member.isfile() or member.isdir()):
                continue
            data = tf.extractfile(member).read() if member.isfile() else b""
            if member.name.startswith("environment/seeds/"):
                seed = copy.copy(member)
                seed.name = "./" + member.name[len("environment/seeds/") :]
                seeds.append((seed, data))
            elif member.isfile():
                files[member.name] = data

    setup_commands = [f"mkdir -p {WORKSPACE} {OUTPUT_DIR}"]
    if seeds:
        payload = _b64_tar(seeds)
        if len(payload) > max_inline_bytes:
            return None
        setup_commands.append(f"printf %s {payload} | base64 -d | tar -xzf - -C {WORKSPACE}")

    return {
        "data_source": DATA_SOURCE,
        "agent_name": "mini_swe_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": files["instruction.md"].decode()},
        ],
        "ability": "terminal",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "task_id": name,
            "sandbox": {
                "image": image,
                "cwd": WORKSPACE,
                # Expected outputs were recorded in UTC with a UTF-8 locale (e.g. ‘quoted’ names in
                # coreutils messages); without these the host's time zone and the C locale leak in.
                "env": SANDBOX_ENV,
                "writable_paths": [OUTPUT_DIR],
                "setup_commands": setup_commands,
                "eval_command": _eval_command(files["tests/expected_output.txt"]),
            },
            "reference_solution": files["solution/solve.sh"].decode(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks_parquet", required=True, help="tasks.parquet from OpenThoughts-Agent-v1-RL")
    parser.add_argument("--local_save_dir", default="~/data/openthoughts_agent_rl")
    parser.add_argument(
        "--image",
        default="openthoughts_agent_rl.sif",
        help="Sandbox image: a .sif name resolved under the tool's image_root, or a docker image",
    )
    parser.add_argument("--test_size", type=int, default=32)
    parser.add_argument("--max_inline_bytes", type=int, default=DEFAULT_MAX_INLINE_BYTES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, skipped = [], []
    for record in pq.read_table(args.tasks_parquet).to_pylist():
        row = convert_task(record["path"], record["task_binary"], args.image, args.max_inline_bytes)
        (rows if row is not None else skipped).append(row or record["path"])
    if skipped:
        print(f"Skipped {len(skipped)} tasks whose seed files are too large to inline: {skipped}")

    random.Random(args.seed).shuffle(rows)
    splits = {"test": rows[: args.test_size], "train": rows[args.test_size :]}
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    for split, split_rows in splits.items():
        for index, row in enumerate(split_rows):
            row["extra_info"] = {"split": split, "index": index, **row["extra_info"]}
        datasets.Dataset.from_list(split_rows).to_parquet(os.path.join(local_save_dir, f"{split}.parquet"))
        print(f"Saved {len(split_rows)} {split} tasks to {local_save_dir}/{split}.parquet")


if __name__ == "__main__":
    main()
