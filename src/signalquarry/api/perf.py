# SPDX-License-Identifier: Apache-2.0
"""``sqy perf publish``: a sanitized live paper-account feed for families that allow it."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from signalquarry._internal.canonical import file_sha256
from signalquarry._internal.contracts.publication import load_publication, publication_path
from signalquarry._internal.paper.models import PaperError
from signalquarry.api import paper as paper_api
from signalquarry.api.envelope import Envelope


def perf_publish(alias: str, *, out: Path, project: Path | None = None) -> Envelope:
    context = paper_api.deployment_context("perf publish", alias, project)
    if isinstance(context, Envelope):
        return context
    family = context.deployment.spec.family
    path = publication_path(context.root, family)
    try:
        publication = load_publication(path) if path is not None else None
    except (ValidationError, ValueError) as exc:
        return Envelope(
            command="perf publish",
            status="invalid",
            reason_codes=["PUBLICATION_INVALID"],
            summary=str(exc)[:2000],
        )
    if publication is None or publication.live_feed != "allow":
        return Envelope(
            command="perf publish",
            status="blocked",
            reason_codes=["PERF_PUBLISH_DENIED"],
            summary=f"family {family} does not allow a live feed (publication policy live_feed: allow; never for commercial families)",
        )
    target = out / alias / "latest.json"
    previous = None
    if target.is_file():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))["snapshot_hash"]
        except (ValueError, KeyError, TypeError):
            return Envelope(
                command="perf publish",
                status="invalid",
                reason_codes=["PERF_FEED_INVALID"],
                summary=f"{target} exists but is not a snapshot; move it aside to start a new chain",
            )
    try:
        document = paper_api.kernel_for(context).snapshot(previous_snapshot_hash=previous)
    except PaperError as exc:
        return paper_api.failure_envelope("perf publish", alias, exc)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Envelope(
        command="perf publish",
        summary=f"{alias} snapshot {document['snapshot_hash'][:19]}… written to {target}",
        data={
            "path": str(target),
            "snapshot_hash": document["snapshot_hash"],
            "previous_snapshot_hash": previous,
            "sessions": len(document["equity_history"]),
        },
        artifacts=[{"path": str(target), "sha256": file_sha256(target), "kind": "performance_snapshot"}],
        evidence={"grade": "paper", "claim_level": "none"},
    )


def perf_capture(alias: str, *, project: Path | None = None) -> Envelope:
    """Record a private, hash-chained performance snapshot under ``paper/<alias>/captures/``.

    Needs no publication policy: nothing leaves the project. Commercial families use
    captures (with the journal) as their paper record for data rooms (NDA export).
    """
    context = paper_api.deployment_context("perf capture", alias, project)
    if isinstance(context, Envelope):
        return context
    directory = context.deployment.state_dir / "captures"
    existing = sorted(directory.glob("*.json")) if directory.is_dir() else []
    previous = None
    if existing:
        try:
            previous = json.loads(existing[-1].read_text(encoding="utf-8"))["snapshot_hash"]
        except (ValueError, KeyError, TypeError):
            return Envelope(
                command="perf capture",
                status="invalid",
                reason_codes=["PERF_FEED_INVALID"],
                summary=f"{existing[-1]} is not a snapshot; the capture chain is broken",
            )
    try:
        document = paper_api.kernel_for(context).snapshot(previous_snapshot_hash=previous)
    except PaperError as exc:
        return paper_api.failure_envelope("perf capture", alias, exc)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = directory / f"{stamp}-{document['snapshot_hash'].split(':', 1)[1][:12]}.json"
    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Envelope(
        command="perf capture",
        summary=f"{alias} capture {len(existing) + 1}: {document['snapshot_hash'][:19]}… (private)",
        data={
            "path": str(target),
            "snapshot_hash": document["snapshot_hash"],
            "previous_snapshot_hash": previous,
            "captures": len(existing) + 1,
        },
        artifacts=[{"path": str(target), "sha256": file_sha256(target), "kind": "performance_capture"}],
        evidence={"grade": "paper", "claim_level": "none"},
    )
