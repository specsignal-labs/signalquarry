# SPDX-License-Identifier: Apache-2.0
"""The JSON envelope every command returns (schema ``signalquarry.cli/v1``)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from signalquarry import __version__
from signalquarry._internal.canonical import file_sha256

ENVELOPE_SCHEMA = "signalquarry.cli/v1"
MAX_DATA_BYTES = 64 * 1024

Status = Literal["ok", "error", "blocked", "usage", "invalid", "unavailable", "busy", "disabled"]
EXIT_CODES: dict[str, int] = {
    "ok": 0,
    "error": 1,
    "blocked": 2,
    "usage": 64,
    "invalid": 65,
    "unavailable": 69,
    "busy": 75,
    "disabled": 78,
}


@dataclass
class Envelope:
    command: str
    status: Status = "ok"
    summary: str = ""
    reason_codes: list[str] = field(default_factory=list[str])
    data: dict[str, Any] = field(default_factory=dict[str, Any])
    metrics: dict[str, Any] = field(default_factory=dict[str, Any])
    evidence: dict[str, Any] | None = None
    artifacts: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    warnings: list[str] = field(default_factory=list[str])
    next_actions: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.monotonic, repr=False)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ENVELOPE_SCHEMA,
            "command": self.command,
            "framework_version": __version__,
            "run_id": self.run_id,
            "duration_ms": int((time.monotonic() - self.started) * 1000),
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "summary": self.summary,
            "data": self.data,
            "metrics": self.metrics,
            "evidence": self.evidence,
            "artifacts": list(self.artifacts),
            "warnings": list(self.warnings),
            "next_actions": list(self.next_actions),
        }


def cap_data(envelope: Envelope, directory: Path) -> Envelope:
    """Move ``data`` larger than MAX_DATA_BYTES into an artifact file (``--detail full`` skips this)."""
    text = json.dumps(envelope.data, ensure_ascii=False, sort_keys=True, default=str)
    size = len(text.encode("utf-8"))
    if size <= MAX_DATA_BYTES:
        return envelope
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{envelope.run_id}.data.json"
    path.write_text(text + "\n", encoding="utf-8")
    envelope.artifacts.append({"path": str(path), "sha256": file_sha256(path), "kind": "data"})
    envelope.data = {"moved_to_artifact": str(path), "bytes": size, "keys": sorted(envelope.data)}
    envelope.warnings.append("DATA_MOVED_TO_ARTIFACT")
    return envelope
