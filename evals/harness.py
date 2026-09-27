# SPDX-License-Identifier: Apache-2.0
"""Shared setup: a fresh demo project, a git baseline and a logging `sqy` shim on PATH."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import yaml

TASKS = Path(__file__).parent / "tasks"

SHIM = """#!/bin/sh
# Logs every sqy call (command words and exit code) for scoring, then passes the output through.
out=$("{real}" "$@"); code=$?
printf '%s\\n' "$out"
python3 - "$code" "$@" <<'PY' >> "{log}"
import json, sys
words = [a for a in sys.argv[2:] if not a.startswith("-")]
command = " ".join(words[:2]) if words[:1] in (["spec"], ["paper"], ["data"], ["trials"], ["holdout"]) else " ".join(words[:1])
print(json.dumps({{"command": command, "argv": sys.argv[2:], "exit": int(sys.argv[1])}}))
PY
exit $code
"""


def load_task(name: str) -> dict:
    return yaml.safe_load((TASKS / f"{name}.yaml").read_text(encoding="utf-8"))


def tasks() -> list[str]:
    return sorted(path.stem for path in TASKS.glob("*.yaml"))


def prepare(workdir: Path, sqy: str, package: str) -> tuple[Path, dict[str, str]]:
    """Create ``workdir/project`` with ``sqy init --demo``, commit it, and install the shim."""
    project = workdir / "project"
    subprocess.run(
        [sqy, "--json", "init", str(project), "--demo", "--package", package], check=True, capture_output=True
    )
    (project / ".gitignore").write_text((project / ".gitignore").read_text() + ".eval/\n")
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.name=eval", "-c", "user.email=eval@example.invalid", "commit", "-qm", "baseline"],
    ):
        subprocess.run(["git", *args], cwd=project, check=True)
    shim_dir = project / ".eval" / "bin"
    shim_dir.mkdir(parents=True)
    log = project / ".eval" / "sqy.log"
    for name in ("sqy", "signalquarry"):
        shim = shim_dir / name
        shim.write_text(SHIM.format(real=sqy, log=log))
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    # Isolation: no market-data or paper credentials are reachable from an eval run, so an
    # agent cannot touch any real (paper) account even if it forges an arm token.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("APCA_", "SIGNALQUARRY_", "ALPACA_"))
    }
    env.update(
        PATH=f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        SIGNALQUARRY_CONFIG_DIR=str(workdir / "config"),
        SIGNALQUARRY_CACHE_DIR=str(workdir / "cache"),
    )
    return project, env
