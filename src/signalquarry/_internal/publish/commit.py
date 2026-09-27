# SPDX-License-Identifier: Apache-2.0
"""Commit-reveal records for strategies and paper journals.

A **spec commitment** binds, at freeze time, everything that determines results
without revealing it::

    c_spec = sha256(canonical_v2({spec, params, gates, freeze_hash, dataset_id,
                                   holdout_start, ledger_head, salt}))
    c_code = sha256(canonical_v2({code_tree_hash, salt}))

The 32-byte salt stays outside the repository (``$SIGNALQUARRY_CONFIG_DIR/salts``,
mode 0600), so the public record cannot be brute-forced back to the parameters.
``reveal`` writes the opening (for a data room or a buyer); ``verify`` recomputes
both digests from it.

A **journal commitment** records a paper journal's head (already an unguessable
hash), so no salt is needed.

The record's ``digest`` is what gets timestamped: with the OpenTimestamps client
(``ots stamp``) when it is installed, otherwise the proof is ``pending``. A
timestamp proves the commitment existed at a time; it does not prove any result.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash, to_canonical
from signalquarry._internal.validation.ledger import ChainedLog

RECORD_SCHEMA = "signalquarry.commitment/v1"
REVEAL_SCHEMA = "signalquarry.commitment-reveal/v1"


class CommitError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


def commitments_dir(root: Path, family: str) -> Path:
    return root / "evidence" / "commitments" / family


def commitment_log(root: Path, family: str) -> ChainedLog:
    return ChainedLog(commitments_dir(root, family) / "commitments.jsonl", RECORD_SCHEMA)


def salts_dir(config_dir: Path, project: str) -> Path:
    return config_dir / "salts" / project


def spec_opening(
    *,
    spec_document: dict[str, Any],
    params: dict[str, Any],
    gates: str,
    freeze_hash: str,
    dataset_id: str | None,
    holdout_start: str | None,
    ledger_head: str | None,
    code_tree_hash: str,
    salt: str,
) -> dict[str, Any]:
    return to_canonical(
        {
            "schema": REVEAL_SCHEMA,
            "spec": spec_document,
            "params": params,
            "gates": gates,
            "freeze_hash": freeze_hash,
            "dataset_id": dataset_id,
            "holdout_start": holdout_start,
            "ledger_head": ledger_head,
            "code_tree_hash": code_tree_hash,
            "salt": salt,
        }
    )


def digests(opening: dict[str, Any]) -> tuple[str, str]:
    spec_part = {k: v for k, v in opening.items() if k not in ("schema", "code_tree_hash")}
    return canonical_hash(spec_part), canonical_hash(
        {"code_tree_hash": opening["code_tree_hash"], "salt": opening["salt"]}
    )


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stamp(path: Path) -> str:
    """Timestamp ``path`` with OpenTimestamps if the client is installed; returns the proof state."""
    client = shutil.which("ots")
    if client is None:
        return "pending"
    completed = subprocess.run(
        [client, "stamp", str(path)], capture_output=True, text=True, timeout=120, check=False
    )
    return (
        "submitted"
        if completed.returncode == 0 and path.with_suffix(path.suffix + ".ots").is_file()
        else "pending"
    )


def create_spec_commitment(
    root: Path,
    *,
    project: str,
    config_dir: Path,
    family: str,
    strategy_id: str,
    opening_fields: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    salt = secrets.token_hex(32)
    opening = spec_opening(**opening_fields, salt=salt)
    c_spec, c_code = digests(opening)
    commit_id = f"{strategy_id}-{now.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    _write_private(salts_dir(config_dir, project) / f"{commit_id}.json", opening)
    record = {
        "kind": "spec",
        "id": commit_id,
        "strategy_id": strategy_id,
        "family": family,
        "created_at": now,
        "c_spec": c_spec,
        "c_code": c_code,
        "digest": canonical_hash({"c_spec": c_spec, "c_code": c_code}),
    }
    return _record(root, family, record)


def create_journal_commitment(
    root: Path, *, family: str, alias: str, journal_head: str, entries: int, now: datetime
) -> dict[str, Any]:
    record = {
        "kind": "journal",
        "id": f"{alias}-journal-{now.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "alias": alias,
        "family": family,
        "created_at": now,
        "journal_head": journal_head,
        "journal_entries": entries,
        "digest": canonical_hash({"alias": alias, "journal_head": journal_head, "journal_entries": entries}),
    }
    return _record(root, family, record)


def _record(root: Path, family: str, record: dict[str, Any]) -> dict[str, Any]:
    directory = commitments_dir(root, family)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['id']}.json"
    body = to_canonical({"schema": RECORD_SCHEMA, **record})
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    body["proof"] = stamp(path)
    entry = commitment_log(root, family).append(
        {k: body[k] for k in ("kind", "id", "digest", "created_at", "proof")}
        | {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
    return {"record": body, "path": path, "log_entry": entry}


def reveal(config_dir: Path, project: str, commit_id: str) -> dict[str, Any]:
    path = salts_dir(config_dir, project) / f"{commit_id}.json"
    if not path.is_file():
        raise CommitError("COMMITMENT_SALT_MISSING", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def verify(record: dict[str, Any], opening: dict[str, Any]) -> list[str]:
    problems = []
    if record.get("schema") != RECORD_SCHEMA or record.get("kind") != "spec":
        problems.append("COMMITMENT_RECORD_INVALID")
    if opening.get("schema") != REVEAL_SCHEMA:
        problems.append("COMMITMENT_REVEAL_INVALID")
    if problems:
        return problems
    c_spec, c_code = digests(opening)
    if c_spec != record["c_spec"]:
        problems.append("COMMITMENT_SPEC_MISMATCH")
    if c_code != record["c_code"]:
        problems.append("COMMITMENT_CODE_MISMATCH")
    if record["digest"] != canonical_hash({"c_spec": record["c_spec"], "c_code": record["c_code"]}):
        problems.append("COMMITMENT_DIGEST_MISMATCH")
    return problems
