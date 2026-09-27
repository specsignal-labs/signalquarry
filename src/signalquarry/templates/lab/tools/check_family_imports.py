"""Fail if code in one family imports another family's package.

    python tools/check_family_imports.py

Shared code belongs upstream in SignalQuarry; sideways imports would make a family
impossible to transfer on its own.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    packages = {
        package.name: family.name
        for family in sorted((ROOT / "families").iterdir())
        if (family / "src").is_dir()
        for package in (family / "src").iterdir()
        if package.is_dir() and (package / "__init__.py").is_file()
    }
    problems = []
    for family in sorted((ROOT / "families").iterdir()):
        for path in sorted(family.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for name in names:
                    owner = packages.get(name.split(".")[0])
                    if owner is not None and owner != family.name:
                        problems.append(
                            f"CROSS_FAMILY_IMPORT {path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)} imports {name} ({owner})"
                        )
    for problem in problems:
        print(problem)
    print(f"{len(packages)} family package(s); {len(problems)} cross-family import(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
