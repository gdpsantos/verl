# mini-swe-agent style RL with a sandboxed bash tool

This example trains a model to fix code by running shell commands, following the
design of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent):

- **Bash is the only tool.** Each call runs in a fresh shell (`bash -lc` via
  `docker exec`), so files persist across turns but `cd` and environment variables do not.
- **History is linear.** Every turn is one model call, and observations are appended
  as tool messages. The trained tokens are exactly the tokens the model generated.
- **Observations follow mini-swe-agent's format.** Each command returns
  `<returncode>…</returncode><output>…</output>`. Long output is cut to a head and tail
  with a warning, and a timed-out command is killed and reported.

Unlike mini-swe-agent, the model calls the tool through the native tool-calling
format (`multi_turn.format`) instead of writing a fenced bash block, and the episode
ends when the model replies without a tool call.

## Components

| File | Role |
|---|---|
| `verl/tools/bash_tool.py` | `BashTool`: runs commands in the trajectory's sandbox (`docker`, `singularity` or `local` backend) |
| `verl/experimental/agent_loop/mini_swe_agent_loop.py` | `mini_swe_agent` loop: opens the sandbox, runs `ToolAgentLoop`, evaluates, always removes the sandbox |
| `verl/utils/reward_score/sandbox_eval.py` | Reward: 1 if the evaluation command exited with 0, else 0 |
| `examples/mini_swe_agent/prepare_toy_bugfix_data.py` | Toy dataset: small Python projects with one bug each |

## Episode lifecycle

1. The loop reads `extra_info["sandbox"]` from the sample, starts a sandbox and runs
   its `setup_commands`. A failing setup command raises, because that is a data or
   infrastructure problem and shouldn't count as a zero reward.
2. The model calls `bash` until it answers without a tool call, or until it hits
   `multi_turn.max_assistant_turns` or the response length limit.
3. The loop runs `eval_command` in the same sandbox and stores the result in
   `extra_fields["sandbox_eval"]`. The fields are `evaluated`, `returncode`, `timed_out`,
   `output` (last 2000 characters), `error`, `num_commands` and `num_command_timeouts`.
4. The sandbox is removed, including when generation fails.

The reward function receives `extra_info["sandbox_eval"]`. `default_compute_score`
routes any `data_source` equal to `sandbox_eval` or starting with `sandbox_eval/` to it.
For another data source, set
`reward.custom_reward_function.path=verl/utils/reward_score/sandbox_eval.py`.

## Dataset format

```python
{
    "data_source": "sandbox_eval/my_dataset",
    "agent_name": "mini_swe_agent",
    "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "<task>"}],
    "reward_model": {"style": "rule", "ground_truth": ""},  # unused by the reward
    "extra_info": {
        "sandbox": {
            "image": "python:3.11-slim",      # docker backend; falls back to the tool's default_image
            "cwd": "/workspace",              # docker backend; falls back to default_cwd
            "setup_commands": ["..."],        # run before the episode
            "eval_command": "...",            # exit code 0 means solved
            # optional: "env": {"KEY": "value"}, "eval_timeout": 900
        },
    },
}
```

`sandbox` may also be a JSON string. That's useful when samples have different keys,
or when `env` would be an empty dict, which parquet can't store.

**Write eval commands that can't be gamed.** Restore the test files before running them
(the toy dataset rewrites them inside `eval_command`), and check the expected tests
explicitly instead of trusting the overall exit code of a test run the model could
have edited.

## Running

```bash
python3 examples/mini_swe_agent/prepare_toy_bugfix_data.py --local_save_dir ~/data/mini_swe_agent_toy
docker pull python:3.11-slim
bash examples/mini_swe_agent/run_qwen3_4b_toy_bugfix.sh
```

Requirements and settings that matter:

- **Docker access.** The user running the Ray workers needs to be able to run
  `docker run`. To use Podman, set `docker_executable: podman`. Without Docker, use the
  Singularity backend (next section). The `local` backend runs the model's commands
  directly on the host and is only for debugging.
- **`multi_turn.max_tool_response_length`** has to be larger than the tool's
  `max_output_chars` (plus about 600 characters of warning text). Otherwise the loop
  truncates observations a second time. The loop's default of 256 is far too small
  for this task.
- **Concurrency.** Each rollout holds one container for its whole episode, so up to
  `train_batch_size × rollout.n` containers can run at the same time. Size the Docker
  host for that. `max_concurrent_starts` limits how many containers each worker starts
  at once.
- **Resource limits.** `docker_run_args` in the example config sets `--network=none`,
  CPU and memory limits. Loosen them only if your tasks need it.

## Clusters without Docker: Singularity / Apptainer

Use [`config/bash_tool_config_singularity.yaml`](config/bash_tool_config_singularity.yaml)
(`TOOL_CONFIG=examples/mini_swe_agent/config/bash_tool_config_singularity.yaml`).
It works with SingularityCE 3.x/4.x, and with Apptainer if you set
`singularity_executable: apptainer`.

How it differs from the Docker backend:

- **One `singularity exec` per command, no long-running container.** Each command
  runs with `--containall --cleanenv`, so it gets a private `/tmp` and home, and none of
  the trainer's environment variables (such as API tokens). Nothing outlives a command,
  so a crashed worker leaves nothing running.
- **Only `cwd` is writable and persistent.** It's a per-trajectory host directory
  mounted at `cwd`. At session start it's filled with whatever the image has at that
  path, for example a repository at `/testbed`, so the model's edits persist. Writes
  anywhere else, such as `pip install` into the image's Python, fail because the image
  is read-only.
- **Images are `.sif` files.** Build them where there is internet access and copy them
  to the cluster:

  ```bash
  singularity build python_3.11-slim.sif docker://python:3.11-slim
  # or from an image saved with `docker save`:
  singularity build task.sif docker-archive://task.tar
  ```

  A sample's `image` can be an absolute `.sif` path, or a Docker-style name that resolves
  under `image_root`. For example, `python:3.11-slim` becomes `<image_root>/python_3.11-slim.sif`,
  so the same dataset works with both backends.

Before launching, make sure Ray workers can run the CLI. `module load singularity`
before `ray start`, or inside the job script, puts it on the workers' `PATH`. You can
also set `singularity_executable` to an absolute path. To check the setup on a compute
node:

```bash
mkdir -p /tmp/w && singularity --silent exec --containall --cleanenv --pwd /workspace \
    --bind /tmp/w:/workspace python_3.11-slim.sif bash -lc 'pwd && python3 --version'
```

If that fails with an error about the mount point `/workspace`, the site configuration
doesn't allow creating mount points that are missing from the image. Use a `cwd` that
exists in the image instead (e.g. `/srv` or `/opt`).

## Using real SWE datasets

For SWE-bench-style data (SWE-Gym, R2E-Gym, SWE-smith), set `image` to the task's
prebuilt environment image, `cwd` to the repository path inside it, and
`eval_command` to the dataset's test command for the `FAIL_TO_PASS` and `PASS_TO_PASS`
tests. Those images are several GB each, so pre-pull them on every node, and expect
much longer trajectories: plan for a `MAX_RESPONSE_LENGTH` of 32k or more.
