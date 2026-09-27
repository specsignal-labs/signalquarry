# SPDX-License-Identifier: Apache-2.0
"""``sqy commit create|reveal|verify``: salted commitments to a frozen strategy and paper journals."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from signalquarry._internal.canonical import file_sha256
from signalquarry._internal.data.credentials import config_dir
from signalquarry._internal.publish.commit import (
    CommitError,
    create_journal_commitment,
    create_spec_commitment,
    reveal,
    verify,
)
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.evaluate import GATES_VERSION
from signalquarry.api.envelope import Envelope
from signalquarry.api.project import load_project


def commit_create(strategy_id: str, *, alias: str | None = None, project: Path | None = None) -> Envelope:
    loaded = load_project(project, "commit create")
    if isinstance(loaded, Envelope):
        return loaded
    root, strategies = loaded
    strategy = strategies.get(strategy_id)
    if strategy is None:
        return Envelope(
            command="commit create",
            status="invalid",
            reason_codes=["STRATEGY_NOT_FOUND"],
            summary=strategy_id,
        )
    now = datetime.now(UTC)
    family = strategy.spec.family
    try:
        if alias is not None:
            journal = ledger.ChainedLog(root / "paper" / alias / "journal.jsonl", "any").entries()
            if not journal:
                return Envelope(
                    command="commit create",
                    status="invalid",
                    reason_codes=["PAPER_CONFIG_NOT_FOUND"],
                    summary=f"no journal for {alias}",
                )
            made = create_journal_commitment(
                root,
                family=family,
                alias=alias,
                journal_head=journal[-1]["hash"],
                entries=len(journal),
                now=now,
            )
        else:
            freeze = ledger.latest_freeze(root, strategy_id)
            if freeze is None or freeze["configuration_hash"] != strategy.configuration_hash:
                return Envelope(
                    command="commit create",
                    status="blocked",
                    reason_codes=["FREEZE_REQUIRED" if freeze is None else "FREEZE_STALE"],
                    summary="commit to a frozen configuration: run `sqy spec freeze` first",
                )
            made = create_spec_commitment(
                root,
                project=root.name,
                config_dir=config_dir(),
                family=family,
                strategy_id=strategy_id,
                opening_fields={
                    "spec_document": strategy.spec.outcome_document(),
                    "params": strategy.params.model_dump(mode="json"),
                    "gates": GATES_VERSION,
                    "freeze_hash": freeze["freeze_hash"],
                    "dataset_id": freeze.get("dataset_id"),
                    "holdout_start": freeze.get("holdout_start"),
                    "ledger_head": ledger.trial_summary(root, family)["head"],
                    "code_tree_hash": strategy.code_tree_hash,
                },
                now=now,
            )
    except (CommitError, ledger.LedgerError) as exc:
        return Envelope(command="commit create", status="blocked", reason_codes=[exc.code], summary=str(exc))
    record, path = made["record"], made["path"]
    envelope = Envelope(
        command="commit create",
        summary=f"{record['kind']} commitment {record['id']} · digest {record['digest'][:19]}… · proof {record['proof']}",
        data={
            "id": record["id"],
            "digest": record["digest"],
            "proof": record["proof"],
            "path": str(path.relative_to(root)),
        },
        artifacts=[{"path": str(path.relative_to(root)), "sha256": file_sha256(path), "kind": "commitment"}],
    )
    if record["proof"] == "pending":
        envelope.warnings.append("COMMITMENT_PROOF_PENDING")
        envelope.next_actions.append(
            {
                "command": f"pip install 'signalquarry[ots]' && ots stamp {path.relative_to(root)}",
                "why": "Timestamp the commitment with OpenTimestamps.",
            }
        )
    if record["kind"] == "spec":
        envelope.next_actions.append(
            {
                "command": "back up $SIGNALQUARRY_CONFIG_DIR/salts",
                "why": "Without the salt the commitment can never be opened.",
            }
        )
    return envelope


def commit_reveal(commit_id: str, *, out: Path, project: Path | None = None) -> Envelope:
    loaded = load_project(project, "commit reveal")
    if isinstance(loaded, Envelope):
        return loaded
    root, _ = loaded
    try:
        opening = reveal(config_dir(), root.name, commit_id)
    except CommitError as exc:
        return Envelope(
            command="commit reveal", status="unavailable", reason_codes=[exc.code], summary=str(exc)
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(opening, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out.chmod(0o600)
    return Envelope(
        command="commit reveal",
        summary=f"opening of {commit_id} written to {out} (private: share only under NDA)",
        data={"path": str(out)},
        warnings=["COMMITMENT_REVEAL_IS_PRIVATE"],
    )


def commit_verify(record_path: Path, reveal_path: Path) -> Envelope:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        opening = json.loads(reveal_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Envelope(
            command="commit verify",
            status="invalid",
            reason_codes=["COMMITMENT_RECORD_INVALID"],
            summary=str(exc),
        )
    problems = verify(record, opening)
    proof = record_path.with_suffix(record_path.suffix + ".ots")
    return Envelope(
        command="commit verify",
        status="ok" if not problems else "blocked",
        reason_codes=problems,
        summary=(
            f"{record.get('id')}: the opening matches the commitment"
            if not problems
            else f"{record.get('id')}: the opening does not match"
        ),
        data={
            "id": record.get("id"),
            "digest": record.get("digest"),
            "opentimestamps_proof": str(proof) if proof.is_file() else None,
            "note": "Check the timestamp with `ots verify` on the .ots file; it proves when the commitment existed, not any result.",
        },
    )
