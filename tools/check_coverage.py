# SPDX-License-Identifier: Apache-2.0
"""Fail unless the modules the evidence depends on keep >= 95% line+branch coverage.

uv run coverage run -m pytest && uv run coverage json -q && uv run python tools/check_coverage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CRITICAL = (
    "src/signalquarry/_internal/canonical.py",
    "src/signalquarry/_internal/engine/",
    "src/signalquarry/_internal/validation/",
    "src/signalquarry/_internal/factors/",
    "src/signalquarry/_internal/data/universe_build.py",
    "src/signalquarry/_internal/paper/runner.py",
    "src/signalquarry/_internal/paper/journal.py",
    "src/signalquarry/_internal/paper/lease.py",
    "src/signalquarry/_internal/paper/arm.py",
    "src/signalquarry/_internal/paper/options_runner.py",
    "src/signalquarry/_internal/options/",
)
THRESHOLD = 95.0


def main() -> int:
    report = json.loads(Path("coverage.json").read_text(encoding="utf-8"))
    failures = []
    for name, data in sorted(report["files"].items()):
        if name.startswith(CRITICAL):
            percent = data["summary"]["percent_covered"]
            status = "ok" if percent >= THRESHOLD else "FAIL"
            print(f"{percent:6.1f}%  {status}  {name}")
            if percent < THRESHOLD:
                failures.append(name)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
