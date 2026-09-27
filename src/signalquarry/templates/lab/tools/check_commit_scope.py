"""Fail if any commit in a range touches more than one family.

    python tools/check_commit_scope.py origin/main..HEAD

A family's history must be extractable on its own (tools/extract_family.sh), so
changes to two families never share a commit. Root files may change with one family.
"""

from __future__ import annotations

import subprocess
import sys


def families_touched(commit: str) -> set[str]:
    names = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", commit],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return {name.split("/")[1] for name in names if name.startswith("families/") and name.count("/") >= 2}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    commits = subprocess.run(
        ["git", "rev-list", argv[1]], capture_output=True, text=True, check=True
    ).stdout.split()
    bad = {c[:12]: sorted(f) for c in commits if len(f := families_touched(c)) > 1}
    for commit, families in bad.items():
        print(f"COMMIT_TOUCHES_SEVERAL_FAMILIES {commit}: {', '.join(families)}")
    print(f"{len(commits)} commit(s) checked; {len(bad)} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
