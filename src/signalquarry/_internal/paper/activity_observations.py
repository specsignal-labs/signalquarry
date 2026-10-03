# SPDX-License-Identifier: Apache-2.0
"""Offline, non-executable observations of private paper activity pages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash, hash_without, to_canonical
from signalquarry._internal.paper.activity_capture import (
    _private_dir,
    _read_private,
    _write_immutable,
    verified_activity_pages,
)
from signalquarry._internal.paper.brokers.alpaca_paper import BrokerActivityPage
from signalquarry._internal.paper.models import PaperError

OBSERVATION_SCHEMA = "signalquarry.paper-activity-observations/v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")


def make_activity_observations(capture: dict[str, Any], pages: list[BrokerActivityPage]) -> dict[str, Any]:
    """Retain provider fields as observations without inferring event semantics."""
    observations = []
    for page in pages:
        for row in page.rows:
            observations.append(
                {
                    "activity_id": row["id"],
                    "activity_type": row.get("activity_type")
                    if isinstance(row.get("activity_type"), str)
                    else None,
                    "symbol": row.get("symbol") if isinstance(row.get("symbol"), str) else None,
                    "reported_date": row.get("date") if isinstance(row.get("date"), str) else None,
                    "reported_transaction_time": row.get("transaction_time")
                    if isinstance(row.get("transaction_time"), str)
                    else None,
                    "fields": sorted(row),
                    "raw_page_sha256": page.sha256,
                    "raw_row_hash": canonical_hash(row),
                }
            )
    record = to_canonical(
        {
            "schema": OBSERVATION_SCHEMA,
            "capture_hash": capture["capture_hash"],
            "account_sha256": capture["account_sha256"],
            "created_after": capture["created_after"],
            "created_until": capture["created_until"],
            "observed_at": capture["observed_at"],
            "normalization_version": 1,
            "observations": observations,
            "redistributable": False,
            "event_links_verified": False,
            "economic_terms_verified": False,
        }
    )
    record["observation_hash"] = canonical_hash(record)
    return record


def observe_activity_capture(root: Path, capture_hash: str, *, account_sha256: str) -> dict[str, Any]:
    """Create an immutable private observation from an already captured window."""
    capture, pages = verified_activity_pages(root, capture_hash, account_sha256=account_sha256)
    try:
        record = make_activity_observations(capture, pages)
    except (TypeError, ValueError, KeyError) as exc:
        raise PaperError("DATA_MANIFEST_INVALID", "blocked", "paper activity observations") from exc
    digest = record["observation_hash"].removeprefix("sha256:")
    body = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    _write_immutable(root / "observations" / f"{digest}.json", body)
    verify_activity_observations(root, record["observation_hash"], account_sha256=account_sha256)
    return record


def verify_activity_observations(root: Path, observation_hash: str, *, account_sha256: str) -> dict[str, Any]:
    """Reconstruct a private observation from verified captured pages."""
    if not isinstance(observation_hash, str) or not _HASH.fullmatch(observation_hash):
        raise PaperError("DATA_MANIFEST_INVALID", "blocked", "activity observation hash")
    _private_dir(root)
    _private_dir(root / "observations")
    try:
        digest = observation_hash.removeprefix("sha256:")
        record = json.loads(_read_private(root / "observations" / f"{digest}.json"))
        if (
            not isinstance(record, dict)
            or record.get("schema") != OBSERVATION_SCHEMA
            or record.get("observation_hash") != observation_hash
            or record.get("account_sha256") != account_sha256
            or hash_without(record, "observation_hash") != observation_hash
            or not isinstance(record.get("capture_hash"), str)
        ):
            raise ValueError("activity observation identity")
        capture, pages = verified_activity_pages(root, record["capture_hash"], account_sha256=account_sha256)
        if make_activity_observations(capture, pages) != record:
            raise ValueError("activity observation contents")
        return record
    except PaperError as exc:
        if exc.code == "PAPER_ACTIVITY_CACHE_UNSAFE":
            raise
        raise PaperError("DATA_MANIFEST_INVALID", "blocked", "paper activity observations") from exc
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise PaperError("DATA_MANIFEST_INVALID", "blocked", "paper activity observations") from exc


def compare_activity_observations(
    root: Path, left_hash: str, right_hash: str, *, account_sha256: str
) -> dict[str, Any]:
    """Compare two verified captures of one window without returning private IDs."""
    left = verify_activity_observations(root, left_hash, account_sha256=account_sha256)
    right = verify_activity_observations(root, right_hash, account_sha256=account_sha256)
    if (left["created_after"], left["created_until"]) != (
        right["created_after"],
        right["created_until"],
    ):
        raise PaperError("USAGE_INVALID", "invalid", "activity observation windows differ")
    left_rows = {item["activity_id"]: item["raw_row_hash"] for item in left["observations"]}
    right_rows = {item["activity_id"]: item["raw_row_hash"] for item in right["observations"]}
    shared = left_rows.keys() & right_rows.keys()
    return {
        "left_observation_hash": left_hash,
        "right_observation_hash": right_hash,
        "created_after": left["created_after"],
        "created_until": left["created_until"],
        "only_left": len(left_rows.keys() - right_rows.keys()),
        "only_right": len(right_rows.keys() - left_rows.keys()),
        "changed": sum(left_rows[key] != right_rows[key] for key in shared),
        "unchanged": sum(left_rows[key] == right_rows[key] for key in shared),
        "redistributable": False,
        "economic_terms_verified": False,
        "event_links_verified": False,
    }
