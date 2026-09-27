# SPDX-License-Identifier: Apache-2.0
"""Every Python file carries an SPDX license header (templates excepted: see REUSE.toml).

python tools/check_headers.py [--fix]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# REUSE-IgnoreStart
HEADER = "# SPDX-License-Identifier: Apache-2.0\n"
MARKER = "SPDX-License-Identifier:"
# REUSE-IgnoreEnd
SCOPES = ("src", "tests", "tools", "evals")
EXEMPT = (
    "src/signalquarry/templates/",
    "tools/leak_scan.py",
)  # leak_scan.py mirrors its private source byte for byte


def files() -> list[Path]:
    found = []
    for scope in SCOPES:
        for path in sorted((ROOT / scope).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if not relative.startswith(EXEMPT) and "__pycache__" not in relative:
                found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args(argv)
    missing = []
    for path in files():
        text = path.read_text(encoding="utf-8")
        if MARKER in "".join(text.splitlines(keepends=True)[:3]):
            continue
        missing.append(path.relative_to(ROOT).as_posix())
        if args.fix:
            path.write_text(HEADER + text, encoding="utf-8")
    if missing and not args.fix:
        print("missing SPDX header:", *missing, sep="\n  ")
        return 1
    print(f"{len(missing)} header(s) added" if args.fix else "all files have SPDX headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
