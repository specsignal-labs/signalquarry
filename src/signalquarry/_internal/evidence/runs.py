# SPDX-License-Identifier: Apache-2.0
"""Run directories: ``.signalquarry/runs/<run_id>/`` with machine-readable artifacts."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash, canonical_json, file_sha256, to_canonical


def new_run_id(configuration_hash: str, now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ") + "-" + configuration_hash.removeprefix("sha256:")[:8]


def unique_run_id(root: Path, run_id: str) -> str:
    """``run_id``, or ``run_id-2``, ``-3``… when a run with that id already exists."""
    runs, candidate, number = root / ".signalquarry" / "runs", run_id, 1
    while (runs / candidate).exists():
        number += 1
        candidate = f"{run_id}-{number}"
    return candidate


def write_run(root: Path, run_id: str, files: dict[str, str | bytes]) -> list[dict[str, str]]:
    run_dir = root / ".signalquarry" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for name, content in files.items():
        path = run_dir / name
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        artifacts.append(
            {"path": str(path.relative_to(root)), "sha256": file_sha256(path), "kind": name.split(".")[0]}
        )
    return artifacts


def csv_text(header: list[str], rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([to_canonical(value) if not isinstance(value, str) else value for value in row])
    return buffer.getvalue()


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def result_document(**fields: Any) -> str:
    body = {"schema": "signalquarry.result/v1", "created_at": datetime.now(UTC), **fields}
    body["result_hash"] = canonical_hash({k: v for k, v in body.items() if k != "created_at"})
    return json.dumps(to_canonical(body), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
