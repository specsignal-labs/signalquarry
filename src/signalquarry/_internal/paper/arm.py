# SPDX-License-Identifier: Apache-2.0
"""``ArmTokenV1``: a human's time-limited permission for one deployment to submit paper orders.

The token binds the alias, strategy configuration, freeze, account and journal head
at arm time. It is stored in ``arm.json`` and its hash is journaled, so editing the
file, changing the strategy, re-freezing or switching accounts disarms the deployment.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash, canonical_json, hash_without
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.models import PaperError

SCHEMA = "signalquarry.arm-token/v1"


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def issue(
    *,
    alias: str,
    strategy_id: str,
    configuration_hash: str,
    freeze_hash: str,
    account_id: str,
    broker: str,
    journal_head: str | None,
    now: datetime,
    days: int,
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "alias": alias,
        "strategy_id": strategy_id,
        "configuration_hash": configuration_hash,
        "freeze_hash": freeze_hash,
        "account_id_sha256": sha256_hex(account_id),
        "broker": broker,
        "journal_head": journal_head,
        "issued_at": now.astimezone(UTC),
        "expires_at": (now + timedelta(days=days)).astimezone(UTC),
    }
    return {**json.loads(canonical_json(body)), "token_hash": canonical_hash(body)}


def write(path: Path, token: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(token, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        token = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return token if isinstance(token, dict) else None


def armed_entry(journal: Journal) -> dict[str, Any] | None:
    """The journal's current arm entry, or ``None`` when disarmed or halted since."""
    last = journal.last("armed", "disarmed", "halted")
    return last if last is not None and last["kind"] == "armed" else None


def check(
    token: dict[str, Any] | None,
    journal: Journal,
    *,
    alias: str,
    configuration_hash: str,
    freeze_hash: str | None,
    account_id: str,
    now: datetime,
) -> dict[str, Any]:
    entry = armed_entry(journal)
    if token is None or entry is None:
        last = journal.last("armed", "disarmed", "halted")
        if last is not None and last["kind"] == "halted":
            raise PaperError("PAPER_HALTED", "disabled", ",".join(last.get("reason_codes", [])))
        raise PaperError("PAPER_NOT_ARMED", "disabled")
    if (
        token.get("schema") != SCHEMA
        or hash_without(token, "token_hash") != token.get("token_hash")
        or entry.get("token_hash") != token["token_hash"]
    ):
        raise PaperError("PAPER_NOT_ARMED", "disabled", "arm token does not match the journal")
    if token["alias"] != alias:
        raise PaperError("PAPER_NOT_ARMED", "disabled", "alias")
    if datetime.fromisoformat(token["expires_at"].replace("Z", "+00:00")) <= now:
        raise PaperError("PAPER_ARM_EXPIRED", "disabled", token["expires_at"])
    if token["configuration_hash"] != configuration_hash or token["freeze_hash"] != freeze_hash:
        raise PaperError("PAPER_ARM_STALE", "disabled", "strategy configuration or freeze changed")
    if token["account_id_sha256"] != sha256_hex(account_id):
        raise PaperError("PAPER_ACCOUNT_MISMATCH", "blocked", "broker account differs from the armed account")
    return token
