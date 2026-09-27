"""Fail if any tracked file mentions a name listed in signalquarry.toml `lab.forbidden_references`.

    python tools/check_references.py

Use it to keep, for example, the showcase repository's name out of the lab: the lab
publishes only through exported bundles, never by linking.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    names = (
        tomllib.loads((ROOT / "signalquarry.toml").read_text()).get("lab", {}).get("forbidden_references", [])
    )
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    hits = []
    for relative in files:
        if relative == "signalquarry.toml":
            continue
        try:
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{relative}: {name}" for name in names if name.lower() in text]
    for hit in hits:
        print(f"FORBIDDEN_REFERENCE {hit}")
    print(f"{len(names)} name(s); {len(hits)} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
