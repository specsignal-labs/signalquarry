# SPDX-License-Identifier: Apache-2.0
"""The deployment journal: an append-only, hash-chained JSONL file and the paper source of truth.

Verified once when opened (under the lease); appends are flushed and fsynced so a
crash never leaves an intent unrecorded. Entry kinds:

``armed`` · ``disarmed`` · ``session_started`` · ``order_intent`` · ``order_submitted``
· ``order_final`` · ``session_completed`` · ``reconciled`` · ``halted`` · ``resumed``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash, canonical_json, to_canonical
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.validation.ledger import ChainedLog, LedgerError

SCHEMA = "signalquarry.paper-journal/v1"


@dataclass
class Journal:
    path: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path) -> Journal:
        try:
            entries = ChainedLog(path, SCHEMA).entries()
        except LedgerError as exc:
            raise PaperError("PAPER_JOURNAL_CORRUPT", "blocked", str(exc)) from exc
        return cls(path, entries)

    @property
    def head(self) -> str | None:
        return self.entries[-1]["hash"] if self.entries else None

    def append(self, kind: str, body: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
        reserved = {"schema", "kind", "at", "seq", "prev", "hash"} & set(body)
        if reserved:
            raise ValueError(f"JOURNAL_FIELD_RESERVED:{','.join(sorted(reserved))}")
        record = to_canonical(
            {
                "schema": SCHEMA,
                "kind": kind,
                "at": (now or datetime.now(UTC)).astimezone(UTC),
                **body,
                "seq": len(self.entries) + 1,
                "prev": self.head,
            }
        )
        entry = {**record, "hash": canonical_hash(record)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.entries.append(entry)
        return entry

    def of_kind(self, *kinds: str) -> list[dict[str, Any]]:
        return [entry for entry in self.entries if entry["kind"] in kinds]

    def last(self, *kinds: str) -> dict[str, Any] | None:
        return next((entry for entry in reversed(self.entries) if entry["kind"] in kinds), None)
