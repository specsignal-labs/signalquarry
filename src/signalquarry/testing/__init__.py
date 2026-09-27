# SPDX-License-Identifier: Apache-2.0
"""Test helpers for strategy projects (provisional in 0.x).

- :func:`conformance` — assert that strategies pass the same checks as ``sqy check``.
- :class:`FakeBroker` — an in-memory paper broker driven by a dataset (``opg`` fills at the open).
- :func:`synthetic_dataset` — deterministic synthetic market data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.paper.brokers.fake import FakeBroker


def conformance(project: Path | str | None = None, strategy: str | None = None) -> list[dict[str, Any]]:
    """Run ``sqy check`` in-process; raise ``AssertionError`` listing every failed check."""
    from signalquarry.api import check

    envelope = check(strategy, project=Path(project) if project is not None else None)
    rows = envelope.data.get("strategies", [])
    failures = [
        f"{row.get('strategy')}: {item['name']} — {item.get('detail', '')}"
        for row in rows
        for item in row.get("checks", [])
        if not item["ok"]
    ]
    if envelope.status != "ok":
        raise AssertionError(
            f"sqy check {envelope.status} {envelope.reason_codes}: "
            + ("; ".join(failures) or envelope.summary)
        )
    return rows


__all__ = ["FakeBroker", "conformance", "synthetic_dataset"]
