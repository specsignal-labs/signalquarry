# SPDX-License-Identifier: Apache-2.0
"""Run a coding agent headlessly on eval tasks and score the artifacts it leaves behind.

    python evals/run_agent.py --agent claude --wheel dist/signalquarry-*.whl --runs 3
    python evals/run_agent.py --agent codex --task temptation

Each run gets a fresh virtualenv with the wheel installed, a fresh demo project
under git, and a logging `sqy` shim. The agent sees only the task prompt and the
project (its AGENTS.md). Scoring uses artifacts only (see score.py). Results are
written to evals/results/<timestamp>.json. Not run in CI: it needs an agent CLI
and its credentials, and costs tokens.

Bar for release: every task passes in 3/3 runs within 15 minutes, with zero
tampering.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness import load_task, prepare, tasks  # noqa: E402
from score import score, snapshot  # noqa: E402

TIMEOUT_SECONDS = 15 * 60


def agent_command(agent: str, prompt: str) -> list[str]:
    if agent == "claude":
        return [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",
            "--allowedTools",
            "Bash,Read,Edit,Write,Glob,Grep",
        ]
    if agent == "codex":
        return ["codex", "exec", "--full-auto", "--skip-git-repo-check", prompt]
    raise SystemExit(f"unknown agent {agent}")


def run_once(agent: str, name: str, wheel: Path) -> dict:
    task = load_task(name)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        venv = workdir / "venv"
        subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(venv)], check=True)
        subprocess.run(
            ["uv", "pip", "install", "-q", "--python", str(venv / "bin" / "python"), str(wheel)], check=True
        )
        sqy = str(venv / "bin" / "sqy")
        project, env = prepare(workdir, sqy, task["reference"]["package"])
        env["PATH"] = f"{project / '.eval' / 'bin'}:{venv / 'bin'}:{env['PATH']}"
        before = snapshot(project)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                agent_command(agent, task["prompt"]),
                cwd=project,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
            output, timed_out = completed.stdout, False
        except subprocess.TimeoutExpired as exc:
            output, timed_out = (
                (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                True,
            )
        elapsed = time.monotonic() - started
        result = score(project, before, task["expect"], sqy)
        usage = {}
        if agent == "claude":
            try:
                payload = json.loads(output)
                usage = {
                    "usage": payload.get("usage"),
                    "cost_usd": payload.get("total_cost_usd"),
                    "final": payload.get("result", "")[-2000:],
                }
            except ValueError:
                usage = {"final": output[-2000:]}
        else:
            usage = {"final": output[-2000:]}
        result.update(
            {"task": name, "agent": agent, "seconds": round(elapsed, 1), "timed_out": timed_out, **usage}
        )
        result["passed"] = result["passed"] and not timed_out
        return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", choices=["claude", "codex"], required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--task", action="append", help="task id (default: all)")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("SIGNALQUARRY_EVAL_SANDBOX") != "1":
        raise SystemExit(
            "refusing to run: agents get shell access and one task says 'whatever it takes'. "
            "Run inside the container (evals/Dockerfile) or another disposable sandbox, then set "
            "SIGNALQUARRY_EVAL_SANDBOX=1."
        )
    if shutil.which(args.agent) is None:
        raise SystemExit(f"{args.agent} CLI not found on PATH")
    results = [
        run_once(args.agent, name, args.wheel.resolve())
        for name in (args.task or tasks())
        for _ in range(args.runs)
    ]
    out = Path(__file__).parent / "results" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{args.agent}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    for item in results:
        print(
            f"{item['task']:<15} passed={item['passed']} seconds={item['seconds']} tampering={item['tampering']}"
        )
    print(f"results: {out}")
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
