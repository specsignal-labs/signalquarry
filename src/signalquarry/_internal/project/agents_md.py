# SPDX-License-Identifier: Apache-2.0
"""Versioned, framework-owned blocks in a project's AGENTS.md.

Each block sits between ``<!-- signalquarry:begin <name> v<N> -->`` and
``<!-- signalquarry:end <name> -->``. Upgrading replaces the blocks with the current
template's text and appends new ones; everything outside the markers is kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BLOCK = re.compile(
    r"<!-- signalquarry:begin (?P<name>[a-z0-9/-]+) v(?P<version>\d+) -->\n"
    r"(?P<body>.*?)"
    r"<!-- signalquarry:end (?P=name) -->\n?",
    re.DOTALL,
)


@dataclass(frozen=True)
class Upgrade:
    text: str
    updated: tuple[str, ...]
    added: tuple[str, ...]
    unchanged: tuple[str, ...]


def blocks(text: str) -> dict[str, re.Match[str]]:
    return {match["name"]: match for match in BLOCK.finditer(text)}


def upgrade(current: str, template: str) -> Upgrade | None:
    """Refresh ``current`` from ``template``; None when ``current`` has no managed blocks."""
    existing = blocks(current)
    if not existing:
        return None
    text = current
    updated, added, unchanged = [], [], []
    for name, fresh in blocks(template).items():
        old = blocks(text).get(name)
        if old is None:
            text = text.rstrip("\n") + "\n\n" + fresh.group(0)
            added.append(name)
        elif old.group(0).rstrip("\n") == fresh.group(0).rstrip("\n"):
            unchanged.append(name)
        else:
            ending = "\n" if old.group(0).endswith("\n") else ""
            text = text[: old.start()] + fresh.group(0).rstrip("\n") + ending + text[old.end() :]
            updated.append(name)
    return Upgrade(
        text if text.endswith("\n") else text + "\n", tuple(updated), tuple(added), tuple(unchanged)
    )
