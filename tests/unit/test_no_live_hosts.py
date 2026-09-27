# SPDX-License-Identifier: Apache-2.0
"""No shipped file names an Alpaca host other than the paper trading API and market data."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
ALLOWED = {"paper-api.alpaca.markets", "data.alpaca.markets"}
HOST = re.compile(r"(?i)\b((?:[a-z0-9-]+\.)*alpaca\.markets)\b")
SCANNED = ("src", "examples", "evals", "docs", "README.md", "AGENTS.md", "mkdocs.yml")


def _files():
    for name in SCANNED:
        path = ROOT / name
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def test_only_paper_and_data_hosts_appear() -> None:
    found = []
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in HOST.finditer(text):
            if match.group(1).lower() not in ALLOWED:
                found.append(f"{path.relative_to(ROOT)}: {match.group(1)}")
    assert found == [], found
