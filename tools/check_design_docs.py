# SPDX-License-Identifier: Apache-2.0
"""Architecture changes must come with a design-doc change.

    python tools/check_design_docs.py --staged            # pre-commit hook
    python tools/check_design_docs.py --range BASE..HEAD  # CI, over a pull request

An *architecture change* is any of:

- a framework module added, removed or renamed under ``src/signalquarry/`` (templates,
  guides and schemas excluded);
- a change to the import-linter contracts (``[tool.importlinter]`` in ``pyproject.toml``).

Such a change passes only if the same change also edits ``docs/design/ARCHITECTURE.md`` or
an ADR under ``docs/adr/``. When a change genuinely needs no design update, acknowledge it
with a commit trailer ``Architecture: unchanged`` (checked in ``--range`` mode) or the
environment variable ``SIGNALQUARRY_ARCH_OK=1`` (for the local hook).

The generated part of ARCHITECTURE.md is checked separately by ``tools/gen_docs.py --check``.
Stdlib only.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib

PACKAGE = "src/signalquarry/"
EXCLUDED = ("src/signalquarry/templates/", "src/signalquarry/guides/", "src/signalquarry/schemas/")
DESIGN = ("docs/design/ARCHITECTURE.md",)
ADR = re.compile(r"^docs/adr/\d{4}-[^/]+\.md$")
TRAILER = re.compile(r"(?im)^Architecture:\s*unchanged\s*$")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _show(spec: str) -> str | None:
    completed = subprocess.run(["git", "show", spec], capture_output=True, text=True, check=False)
    return completed.stdout if completed.returncode == 0 else None


def _contracts(text: str | None) -> object:
    if text is None:
        return None
    try:
        return tomllib.loads(text).get("tool", {}).get("importlinter")
    except tomllib.TOMLDecodeError:
        return "unparseable"


def architecture_changes(name_status: str, old_pyproject: str | None, new_pyproject: str | None) -> list[str]:
    """What in this change counts as an architecture change (empty: nothing)."""
    changes = []
    for line in name_status.splitlines():
        parts = line.split("\t")
        status, paths = parts[0], parts[1:]
        if status[0] not in "ADR":
            continue
        for path in paths:
            if path.startswith(PACKAGE) and path.endswith(".py") and not path.startswith(EXCLUDED):
                verb = {"A": "added", "D": "removed", "R": "renamed"}[status[0]]
                changes.append(f"module {verb}: {path}")
                break
    if _contracts(old_pyproject) != _contracts(new_pyproject):
        changes.append("import-linter contracts changed in pyproject.toml")
    return changes


def documented(name_status: str) -> bool:
    for line in name_status.splitlines():
        for path in line.split("\t")[1:]:
            if path in DESIGN or ADR.match(path):
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="check the staged change (pre-commit)")
    mode.add_argument("--range", help="check a commit range, e.g. origin/main..HEAD (CI)")
    args = parser.parse_args(argv)
    if args.staged:
        name_status = _git("diff", "--cached", "--name-status", "-M")
        old, new = _show("HEAD:pyproject.toml"), _show(":pyproject.toml")
        acknowledged = os.environ.get("SIGNALQUARRY_ARCH_OK") == "1"
    else:
        base, _, head = args.range.partition("..")
        head = head or "HEAD"
        merge_base = _git("merge-base", base, head).strip()
        name_status = _git("diff", "--name-status", "-M", merge_base, head)
        old, new = _show(f"{merge_base}:pyproject.toml"), _show(f"{head}:pyproject.toml")
        acknowledged = bool(TRAILER.search(_git("log", "--format=%B", f"{merge_base}..{head}")))
    changes = architecture_changes(name_status, old, new)
    if not changes:
        print("DESIGN_DOCS_OK no architecture change")
        return 0
    if documented(name_status):
        print(f"DESIGN_DOCS_OK {len(changes)} architecture change(s), design docs updated")
        return 0
    if acknowledged:
        print(f"DESIGN_DOCS_ACKNOWLEDGED {len(changes)} architecture change(s) marked 'unchanged'")
        return 0
    print("DESIGN_DOCS_MISSING: this change alters the architecture but not the design docs:")
    for change in changes:
        print(f"  - {change}")
    print(
        "Update docs/design/ARCHITECTURE.md (and run tools/gen_docs.py) or add an ADR in docs/adr/.\n"
        "If no design update is needed, add the trailer 'Architecture: unchanged' to the commit\n"
        "message (CI) or set SIGNALQUARRY_ARCH_OK=1 for the local hook."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
