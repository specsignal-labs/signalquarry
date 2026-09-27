# SPDX-License-Identifier: Apache-2.0
"""``sqy evidence export``: a verifiable evidence bundle for one family, under its publication policy."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import ValidationError

from signalquarry import __version__
from signalquarry._internal.canonical import file_sha256
from signalquarry._internal.contracts.publication import load_publication, publication_path
from signalquarry._internal.evidence.verify import verify_bundle
from signalquarry._internal.project.project import ProjectError, find_root, load_config, load_strategies
from signalquarry._internal.publish.export import ExportError, StrategyEvidence, build_bundle
from signalquarry._internal.validation.ledger import ChainedLog, LedgerError
from signalquarry.api.envelope import Envelope


def _exports(root: Path) -> ChainedLog:
    return ChainedLog(root / "evidence" / "exports.jsonl", "signalquarry.export/v1")


def evidence_export(
    family: str,
    *,
    tier: str = "public",
    out: Path | None = None,
    project: Path | None = None,
    today: date | None = None,
) -> Envelope:
    envelope = Envelope(command="evidence export")
    try:
        root = find_root(project)
        config = load_config(root)
        strategies = load_strategies(config)
    except ProjectError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            [exc.code],
            exc.detail or exc.code,
        )
        return envelope
    path = publication_path(root, family)
    if path is None:
        envelope.status, envelope.reason_codes = "invalid", ["PUBLICATION_NOT_FOUND"]
        envelope.summary = f"no publication/{family}.publication.yaml (nothing is published by default)"
        return envelope
    try:
        publication = load_publication(path)
    except (ValidationError, ValueError) as exc:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            ["PUBLICATION_INVALID"],
            str(exc)[:2000],
        )
        return envelope
    now = datetime.now(UTC)
    produced_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        previous = [e for e in _exports(root).entries() if e["family"] == family and e["tier"] == tier]
    except LedgerError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = "blocked", [exc.code], str(exc)
        return envelope
    target = out or root / ".signalquarry" / "exports" / f"{family}-{tier}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    try:
        manifest = build_bundle(
            root,
            publication,
            {
                strategy_id: StrategyEvidence(item.spec, item.configuration_hash)
                for strategy_id, item in strategies.items()
            },
            tier=tier,
            out=target,
            produced_at=produced_at,
            today=today or now.date(),
            framework_version=__version__,
            project=config.name,
            supersedes=previous[-1]["bundle_hash"] if previous else None,
        )
    except ExportError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = "invalid", [exc.code], str(exc)
        return envelope
    report = verify_bundle(target, max_tier=tier)
    if not report["valid"]:
        envelope.status, envelope.reason_codes = "error", ["BUNDLE_INVALID"]
        envelope.summary, envelope.data = (
            "the exported bundle failed verification",
            {"errors": report["errors"]},
        )
        return envelope
    _exports(root).append(
        {
            "at": now,
            "family": family,
            "tier": tier,
            "bundle_hash": manifest["bundle_hash"],
            "files": len(manifest["files"]),
            "supersedes": manifest.get("supersedes"),
        }
    )
    envelope.summary = (
        f"{family} {tier} bundle {manifest['bundle_hash'][:19]}… ({len(manifest['files'])} files)"
    )
    envelope.data = {
        "path": str(target),
        "bundle_hash": manifest["bundle_hash"],
        "tier": tier,
        "files": len(manifest["files"]),
        "supersedes": manifest.get("supersedes"),
    }
    envelope.artifacts = [
        {"path": str(target / "bundle.json"), "sha256": file_sha256(target / "bundle.json"), "kind": "bundle"}
    ]
    envelope.next_actions = [
        {
            "command": f"python tools/ingest.py {target} --source <label>",
            "why": "In the showcase checkout: verify and add the bundle, then open a pull request.",
        }
        if tier == "public"
        else {"command": "(data room)", "why": "NDA bundles go only to a per-deal private data room."}
    ]
    return envelope


def bundle_verify(bundle: Path, *, max_tier: str = "public") -> Envelope:
    if not (bundle / "bundle.json").is_file():
        return Envelope(
            command="evidence verify",
            status="invalid",
            reason_codes=["BUNDLE_INVALID"],
            summary=f"no bundle.json in {bundle}",
        )
    report = verify_bundle(bundle, max_tier=max_tier)
    return Envelope(
        command="evidence verify",
        status="ok" if report["valid"] else "blocked",
        reason_codes=[] if report["valid"] else ["BUNDLE_INVALID"],
        summary=f"bundle {report.get('bundle_hash')} {'valid' if report['valid'] else 'invalid'}",
        data=report,
    )
