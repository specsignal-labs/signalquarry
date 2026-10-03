# SPDX-License-Identifier: Apache-2.0
"""Private, immutable captures of read-only Alpaca paper account activities.

Raw account and order material never enters the project or an API envelope.
The creation-time window is not proof of economic effective or settlement time.
These pages are evidence for future provider decoding, not executable terms.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash, hash_without, to_canonical
from signalquarry._internal.paper.brokers.alpaca_paper import BrokerActivityPage
from signalquarry._internal.paper.models import PaperError

CAPTURE_SCHEMA = "signalquarry.paper-activity-capture/v1"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_ENDPOINT = "/v2/account/activities"


def _base_params(created_after: datetime, created_until: datetime) -> dict[str, str]:
    return {
        "after": created_after.astimezone(UTC).isoformat(),
        "until": created_until.astimezone(UTC).isoformat(),
        "direction": "asc",
        "page_size": "100",
    }


def make_activity_capture(
    pages: list[BrokerActivityPage],
    *,
    account_sha256: str,
    created_after: datetime,
    created_until: datetime,
    observed_at: datetime,
) -> dict[str, Any]:
    """Validate complete pagination and describe a capture without raw rows."""
    if (
        not _HASH.fullmatch(account_sha256)
        or any(
            value.tzinfo is None or value.utcoffset() is None
            for value in (created_after, created_until, observed_at)
        )
        or created_after >= created_until
        or created_until > observed_at
        or not pages
    ):
        raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity capture identity")
    base = _base_params(created_after, created_until)
    token: str | None = None
    seen: set[str] = set()
    references: list[dict[str, Any]] = []
    for index, page in enumerate(pages):
        params = dict(base)
        if token is not None:
            params["page_token"] = token
        if page.params != params or hashlib.sha256(page.body).hexdigest() != page.sha256:
            raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity page identity")
        try:
            rows = json.loads(page.body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity page JSON") from exc
        if (
            not isinstance(rows, list)
            or len(rows) > 100
            or any(not isinstance(row, dict) for row in rows)
            or tuple(rows) != page.rows
        ):
            raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity page rows")
        for row in rows:
            activity_id = row.get("id")
            if not isinstance(activity_id, str) or not activity_id or activity_id in seen:
                raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity id")
            seen.add(activity_id)
        if index < len(pages) - 1 and len(rows) != 100:
            raise PaperError("PAPER_ACTIVITY_CAPTURE_INCOMPLETE", "blocked", "pagination stopped early")
        token = rows[-1]["id"] if len(rows) == 100 else None
        references.append({"params": page.params, "sha256": page.sha256, "count": len(rows)})
    if len(pages[-1].rows) == 100:
        raise PaperError("PAPER_ACTIVITY_CAPTURE_INCOMPLETE", "blocked", "terminal page missing")
    capture = to_canonical(
        {
            "schema": CAPTURE_SCHEMA,
            "account_sha256": account_sha256,
            "created_after": created_after.astimezone(UTC),
            "created_until": created_until.astimezone(UTC),
            "observed_at": observed_at.astimezone(UTC),
            "pages": references,
            "activities": len(seen),
            "redistributable": False,
        }
    )
    capture["capture_hash"] = canonical_hash(capture)
    return capture


def _private_dir(path: Path) -> None:
    if path.is_symlink():
        raise PaperError("PAPER_ACTIVITY_CACHE_UNSAFE", "blocked", "cache symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PaperError("PAPER_ACTIVITY_CACHE_UNSAFE", "blocked", "cache permissions")


def capture_root(cache_dir: Path, project_root: Path) -> Path:
    """Reject public-project storage before any account page is requested."""
    resolved_cache = cache_dir.expanduser().resolve()
    if resolved_cache.is_relative_to(project_root.resolve()):
        raise PaperError("PAPER_ACTIVITY_CACHE_UNSAFE", "blocked", "cache inside project")
    path = resolved_cache / "paper-activity-captures"
    _private_dir(path)
    return path


def _write_immutable(path: Path, body: bytes) -> None:
    _private_dir(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".capture-", delete=False) as handle:
        temporary = Path(handle.name)
        os.fchmod(handle.fileno(), 0o600)
        handle.write(body)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = _read_private(path)
            except OSError as exc:
                raise PaperError(
                    "PAPER_ACTIVITY_CACHE_UNSAFE", "blocked", "existing capture unreadable"
                ) from exc
            if existing != body:
                raise PaperError(
                    "PAPER_ACTIVITY_CACHE_UNSAFE", "blocked", "existing capture changed"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _read_private(path: Path) -> bytes:
    if path.is_symlink():
        raise PaperError("PAPER_ACTIVITY_CACHE_UNSAFE", "blocked", "capture symlink")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PaperError("PAPER_ACTIVITY_CACHE_UNSAFE", "blocked", "capture permissions")
    return path.read_bytes()


def store_activity_capture(
    root: Path,
    pages: list[BrokerActivityPage],
    *,
    account_sha256: str,
    created_after: datetime,
    created_until: datetime,
    observed_at: datetime,
) -> dict[str, Any]:
    capture = make_activity_capture(
        pages,
        account_sha256=account_sha256,
        created_after=created_after,
        created_until=created_until,
        observed_at=observed_at,
    )
    for page in pages:
        _write_immutable(root / "pages" / f"{page.sha256}.json", page.body)
    digest = capture["capture_hash"].removeprefix("sha256:")
    encoded = (json.dumps(capture, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    _write_immutable(root / "records" / f"{digest}.json", encoded)
    verify_activity_capture(root, capture["capture_hash"], account_sha256=account_sha256)
    return capture


def verify_activity_capture(root: Path, capture_hash: str, *, account_sha256: str) -> dict[str, Any]:
    """Re-hash every private page and replay the pagination/metadata rules."""
    if not isinstance(capture_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", capture_hash):
        raise PaperError("DATA_MANIFEST_INVALID", "blocked", "activity capture hash")
    _private_dir(root)
    _private_dir(root / "records")
    _private_dir(root / "pages")
    try:
        path = root / "records" / f"{capture_hash.removeprefix('sha256:')}.json"
        record = json.loads(_read_private(path))
        if (
            not isinstance(record, dict)
            or record.get("schema") != CAPTURE_SCHEMA
            or record.get("capture_hash") != capture_hash
            or record.get("account_sha256") != account_sha256
            or hash_without(record, "capture_hash") != capture_hash
            or not isinstance(record.get("pages"), list)
        ):
            raise ValueError("capture identity")
        pages = []
        for ref in record["pages"]:
            if (
                not isinstance(ref, dict)
                or set(ref) != {"params", "sha256", "count"}
                or not isinstance(ref["sha256"], str)
                or not _HASH.fullmatch(ref["sha256"])
                or not isinstance(ref["params"], dict)
            ):
                raise ValueError("page reference")
            body = _read_private(root / "pages" / f"{ref['sha256']}.json")
            if hashlib.sha256(body).hexdigest() != ref["sha256"]:
                raise ValueError("page hash")
            rows = json.loads(body)
            if not isinstance(rows, list) or len(rows) != ref["count"]:
                raise ValueError("page count")
            pages.append(BrokerActivityPage(ref["params"], body, ref["sha256"], tuple(rows)))
        rebuilt = make_activity_capture(
            pages,
            account_sha256=account_sha256,
            created_after=datetime.fromisoformat(record["created_after"].replace("Z", "+00:00")),
            created_until=datetime.fromisoformat(record["created_until"].replace("Z", "+00:00")),
            observed_at=datetime.fromisoformat(record["observed_at"].replace("Z", "+00:00")),
        )
        if rebuilt != record:
            raise ValueError("capture contents")
        return record
    except PaperError as exc:
        if exc.code == "PAPER_ACTIVITY_CACHE_UNSAFE":
            raise
        raise PaperError("DATA_MANIFEST_INVALID", "blocked", "paper activity capture") from exc
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise PaperError("DATA_MANIFEST_INVALID", "blocked", "paper activity capture") from exc


def verified_activity_pages(
    root: Path, capture_hash: str, *, account_sha256: str
) -> tuple[dict[str, Any], list[BrokerActivityPage]]:
    """Read pages only after capture verification, checking hashes again on use."""
    capture = verify_activity_capture(root, capture_hash, account_sha256=account_sha256)
    pages: list[BrokerActivityPage] = []
    try:
        for ref in capture["pages"]:
            body = _read_private(root / "pages" / f"{ref['sha256']}.json")
            if hashlib.sha256(body).hexdigest() != ref["sha256"]:
                raise ValueError("activity page changed after verification")
            rows = json.loads(body)
            if not isinstance(rows, list) or len(rows) != ref["count"]:
                raise ValueError("activity page rows changed")
            pages.append(BrokerActivityPage(ref["params"], body, ref["sha256"], tuple(rows)))
    except PaperError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise PaperError("DATA_MANIFEST_INVALID", "blocked", "paper activity pages") from exc
    return capture, pages
