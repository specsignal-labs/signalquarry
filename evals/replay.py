# SPDX-License-Identifier: Apache-2.0
"""Deterministic replay of each task's reference solution through the real CLI (runs in CI).

    python evals/replay.py --sqy /path/to/venv/bin/sqy

Exercises the same setup and scorer as the agent eval without an agent, so the
harness itself is tested on every change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml  # noqa: E402
from harness import load_task, prepare, tasks  # noqa: E402
from score import score, snapshot  # noqa: E402


def replay(name: str, sqy: str) -> dict:
    task = load_task(name)
    reference = task["reference"]
    with tempfile.TemporaryDirectory() as tmp:
        project, env = prepare(Path(tmp), sqy, reference["package"])
        before = snapshot(project)
        for relative, content in reference["files"].items():
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if reference["modules"]:
            config = project / "signalquarry.toml"
            text = config.read_text()
            modules = yaml.safe_load("[" + text.split("modules = [", 1)[1].split("]", 1)[0] + "]")
            joined = ", ".join(f'"{m}"' for m in [*modules, *reference["modules"]])
            config.write_text(text.replace(text.split("modules = ", 1)[1].split("\n", 1)[0], f"[{joined}]"))
        if reference["files"] or reference["modules"]:
            subprocess.run(["git", "add", "-A"], cwd=project, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=eval",
                    "-c",
                    "user.email=eval@example.invalid",
                    "commit",
                    "-qm",
                    "reference",
                ],
                cwd=project,
                check=True,
            )
        for command in reference["commands"]:
            subprocess.run(
                ["sqy", "--json", *command], cwd=project, env=env, capture_output=True, check=False
            )
        return score(project, before, task["expect"], sqy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqy", required=True, help="path to the installed sqy executable")
    parser.add_argument("--task", action="append", help="task id (default: all)")
    args = parser.parse_args()
    results = {name: replay(name, args.sqy) for name in (args.task or tasks())}
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(result["passed"] for result in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
