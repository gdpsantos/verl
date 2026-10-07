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
Generate a toy bug-fixing dataset for the mini_swe_agent agent loop.

Each task is a tiny Python project with one buggy function, a correct distractor
function and a unittest file. The model fixes the bug with the bash tool; after
the episode the agent loop rewrites the test file (so edits to the tests do not
count) and runs it. The reward is 1 when the tests pass.

Only the Python standard library is needed inside the sandbox, so any image with
python3 and bash works (default: python:3.11-slim).
"""

import argparse
import os
import random
import textwrap

import datasets

DATA_SOURCE = "sandbox_eval/toy_bugfix"
HEREDOC = "VERL_EOF"

SYSTEM_PROMPT = (
    "You are a software engineer working in a Linux sandbox. Use the `bash` tool to inspect "
    "files, run commands and edit code. Each tool call runs in a new shell: files persist between "
    "calls, but `cd` and environment variables do not, so prefix commands with `cd <dir> &&` when "
    "needed. Interactive programs such as editors and pagers are not available; edit files with "
    "`sed`, `cat <<'EOF' > file`, or a short `python3` script. Run one command per tool call. "
    "When the task is done, reply with a short summary of your change and do not call the tool again."
)


def _sum_to(fn):
    buggy = f'''
    def {fn}(n):
        """Return the sum of the integers from 1 to n inclusive."""
        total = 0
        for i in range(1, n):
            total += i
        return total
    '''
    fixed = buggy.replace("range(1, n)", "range(1, n + 1)")
    k = random.randint(6, 40)
    tests = [
        f"self.assertEqual({fn}(1), 1)",
        f"self.assertEqual({fn}(5), 15)",
        f"self.assertEqual({fn}({k}), {k * (k + 1) // 2})",
    ]
    return buggy, fixed, tests


def _is_even(fn):
    buggy = f'''
    def {fn}(x):
        """Return True if x is even."""
        return x % 2 == 1
    '''
    fixed = buggy.replace("== 1", "== 0")
    k = random.randint(1, 50) * 2
    tests = [f"self.assertTrue({fn}({k}))", f"self.assertFalse({fn}({k + 1}))", f"self.assertTrue({fn}(0))"]
    return buggy, fixed, tests


def _max_of(fn):
    buggy = f'''
    def {fn}(values):
        """Return the largest value in a non-empty list."""
        best = 0
        for v in values:
            if v > best:
                best = v
        return best
    '''
    fixed = buggy.replace("best = 0", "best = values[0]")
    a, b = random.randint(2, 9), random.randint(10, 30)
    tests = [f"self.assertEqual({fn}([{a}, {b}, 1]), {b})", f"self.assertEqual({fn}([-{b}, -{a}, -{b + a}]), -{a})"]
    return buggy, fixed, tests


def _count_vowels(fn):
    buggy = f'''
    def {fn}(text):
        """Count the vowels (a, e, i, o, u) in text, ignoring case."""
        return sum(1 for ch in text if ch in "aeiou")
    '''
    fixed = buggy.replace("if ch in", "if ch.lower() in")
    # Every word needs an uppercase vowel, otherwise the buggy version passes.
    word = random.choice(["Apple", "Education", "OUTPUT", "Idea", "Umbrella"])
    expected = sum(1 for ch in word if ch.lower() in "aeiou")
    tests = [f'self.assertEqual({fn}("{word}"), {expected})', f'self.assertEqual({fn}("xyz"), 0)']
    return buggy, fixed, tests


def _factorial(fn):
    buggy = f'''
    def {fn}(n):
        """Return n! for n >= 0."""
        result = 1
        for i in range(2, n):
            result *= i
        return result
    '''
    fixed = buggy.replace("range(2, n)", "range(2, n + 1)")
    k = random.randint(4, 9)
    expected = 1
    for i in range(2, k + 1):
        expected *= i
    tests = [
        f"self.assertEqual({fn}(0), 1)",
        f"self.assertEqual({fn}(3), 6)",
        f"self.assertEqual({fn}({k}), {expected})",
    ]
    return buggy, fixed, tests


def _clamp(fn):
    buggy = f'''
    def {fn}(x, low, high):
        """Clamp x into the closed interval [low, high]."""
        return max(high, min(low, x))
    '''
    fixed = buggy.replace("max(high, min(low, x))", "max(low, min(high, x))")
    lo, hi = random.randint(0, 5), random.randint(10, 20)
    tests = [
        f"self.assertEqual({fn}({lo - 3}, {lo}, {hi}), {lo})",
        f"self.assertEqual({fn}({hi + 7}, {lo}, {hi}), {hi})",
        f"self.assertEqual({fn}({lo + 2}, {lo}, {hi}), {lo + 2})",
    ]
    return buggy, fixed, tests


def _average(fn):
    buggy = f'''
    def {fn}(values):
        """Return the arithmetic mean of a non-empty list of numbers."""
        return sum(values) // len(values)
    '''
    fixed = buggy.replace("//", "/")
    a = random.randint(1, 20)
    tests = [f"self.assertAlmostEqual({fn}([{a}, {a + 1}]), {a + 0.5})", f"self.assertAlmostEqual({fn}([4, 4, 4]), 4)"]
    return buggy, fixed, tests


def _reverse_words(fn):
    buggy = f'''
    def {fn}(sentence):
        """Return the words of sentence in reverse order, separated by single spaces."""
        return sentence[::-1]
    '''
    fixed = buggy.replace("sentence[::-1]", '" ".join(reversed(sentence.split()))')
    words = random.sample(["red", "green", "blue", "cat", "dog", "sun", "moon"], 3)
    tests = [
        f'self.assertEqual({fn}("{" ".join(words)}"), "{" ".join(reversed(words))}")',
        f'self.assertEqual({fn}("one"), "one")',
    ]
    return buggy, fixed, tests


def _dedupe(fn):
    buggy = f'''
    def {fn}(items):
        """Remove duplicates from items, keeping the order of first occurrence."""
        return list(set(items))
    '''
    fixed = buggy.replace("return list(set(items))", "return list(dict.fromkeys(items))")
    tests = [
        f"self.assertEqual({fn}([3, 1, 3, 2, 1]), [3, 1, 2])",
        f'self.assertEqual({fn}(["b", "a", "b"]), ["b", "a"])',
    ]
    return buggy, fixed, tests


TEMPLATES = {
    "sum_to": (_sum_to, ["sum_to", "triangular", "sum_first_n"]),
    "is_even": (_is_even, ["is_even", "even", "check_even"]),
    "max_of": (_max_of, ["max_of", "largest", "find_max"]),
    "count_vowels": (_count_vowels, ["count_vowels", "vowel_count", "num_vowels"]),
    "factorial": (_factorial, ["factorial", "fact", "compute_factorial"]),
    "clamp": (_clamp, ["clamp", "bound", "limit"]),
    "average": (_average, ["average", "mean", "avg"]),
    "reverse_words": (_reverse_words, ["reverse_words", "flip_words", "words_backwards"]),
    "dedupe": (_dedupe, ["dedupe", "unique", "remove_duplicates"]),
}
MODULE_NAMES = ["utils", "helpers", "toolkit", "core", "funcs", "common", "mathlib"]


def _write_file(path: str, content: str) -> str:
    return f"cat > {path} <<'{HEREDOC}'\n{content.rstrip()}\n{HEREDOC}"


def make_task(bug_name: str, distractor_name: str, image: str, cwd: str) -> tuple[dict, str]:
    module = random.choice(MODULE_NAMES)
    bug_fn_name = random.choice(TEMPLATES[bug_name][1])
    distractor_fn_name = random.choice(TEMPLATES[distractor_name][1])
    buggy, _, bug_tests = TEMPLATES[bug_name][0](bug_fn_name)
    _, distractor_fixed, distractor_tests = TEMPLATES[distractor_name][0](distractor_fn_name)

    functions = [textwrap.dedent(buggy).strip(), textwrap.dedent(distractor_fixed).strip()]
    random.shuffle(functions)
    module_code = "\n\n\n".join(functions) + "\n"

    test_methods = []
    for name, asserts in [(bug_fn_name, bug_tests), (distractor_fn_name, distractor_tests)]:
        body = "\n".join(f"        {a}" for a in asserts)
        test_methods.append(f"    def test_{name}(self):\n{body}")
    test_code = (
        "import unittest\n\n"
        f"from {module} import {bug_fn_name}, {distractor_fn_name}\n\n\n"
        f"class Test{module.capitalize()}(unittest.TestCase):\n" + "\n\n".join(test_methods) + "\n\n\n"
        'if __name__ == "__main__":\n    unittest.main()\n'
    )
    readme = f"# {module}\n\nRun the tests with:\n\n    python3 -m unittest -v test_{module}\n"

    test_file = f"test_{module}.py"
    sandbox = {
        "image": image,
        "cwd": cwd,
        "setup_commands": [
            _write_file(f"{module}.py", module_code),
            _write_file(test_file, test_code),
            _write_file("README.md", readme),
        ],
        # Rewrite the tests before running them so the model cannot pass by editing them.
        "eval_command": _write_file(test_file, test_code) + f"\npython3 -m unittest -q test_{module}",
    }
    task = (
        f"The unit tests in `{test_file}` are failing. Find and fix the bug in `{module}.py` so that "
        f"`python3 -m unittest -v test_{module}` passes. Do not modify `{test_file}`."
    )
    return sandbox, task


def build_split(split: str, size: int, image: str, cwd: str) -> datasets.Dataset:
    rows = []
    names = list(TEMPLATES)
    for idx in range(size):
        bug_name, distractor_name = random.sample(names, 2)
        sandbox, task = make_task(bug_name, distractor_name, image, cwd)
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "agent_name": "mini_swe_agent",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ],
                "ability": "coding",
                "reward_model": {"style": "rule", "ground_truth": bug_name},
                "extra_info": {"split": split, "index": idx, "bug": bug_name, "sandbox": sandbox},
            }
        )
    return datasets.Dataset.from_list(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/mini_swe_agent_toy")
    parser.add_argument("--train_size", type=int, default=2048)
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--cwd", default="/workspace")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    build_split("train", args.train_size, args.image, args.cwd).to_parquet(
        os.path.join(local_save_dir, "train.parquet")
    )
    build_split("test", args.test_size, args.image, args.cwd).to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"Saved {args.train_size} train / {args.test_size} test tasks to {local_save_dir}")
