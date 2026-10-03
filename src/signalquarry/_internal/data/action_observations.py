# SPDX-License-Identifier: Apache-2.0
"""Versioned, non-executable observations of Alpaca corporate-action rows.

These records deliberately contain no consideration or successor mapping. They
can be inspected and compared across captures, but cannot drive a ledger or
authorize paper orders. Raw pages remain the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash, hash_without, to_canonical
from signalquarry._internal.data.alpaca import CORPORATE_ACTIONS_PATH, ProviderError, RawPage
from signalquarry._internal.data.library import Library, LibraryError

NORMALIZATION_VERSION = 1
CAPTURE_SCHEMA = "signalquarry.corporate-action-observations/v1"
_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CorporateActionObservation:
    """One provider row seen at a known time, without inferred economic terms."""

    provider_id: str
    kind: str
    symbol: str | None
    effective_date: date | None
    process_date: date
    observed_at: datetime
    page_hashes: tuple[str, ...]
    row_sha256: str
    normalization_version: int = NORMALIZATION_VERSION


def observations_from_pages(
    pages: list[RawPage], *, observed_at: datetime
) -> tuple[CorporateActionObservation, ...]:
    """Extract auditable metadata from one frozen capture of action pages.

    A missing ID or process date cannot support revision tracking. A missing
    ex-date is retained as an incomplete observation, never treated as an
    effective event. A conflicting repeat ID within a capture is rejected.
    """
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action observation time")
    seen: dict[str, CorporateActionObservation] = {}
    for page in pages:
        if page.endpoint != CORPORATE_ACTIONS_PATH or not _HASH.fullmatch(page.sha256):
            raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action page identity")
        if hashlib.sha256(page.body).hexdigest() != page.sha256:
            raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action page hash")
        try:
            payload = json.loads(page.body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action page JSON") from exc
        if not isinstance(payload, dict) or payload != page.payload:
            raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action page payload")
        actions = payload.get("corporate_actions")
        if not isinstance(actions, dict):
            raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate actions collection")
        for kind, rows in actions.items():
            if not isinstance(kind, str) or not isinstance(rows, list):
                raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action category")
            for row in rows:
                if not isinstance(row, dict):
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action row")
                provider_id = row.get("id")
                process = row.get("process_date")
                symbol = row.get("symbol")
                effective = row.get("ex_date")
                if not isinstance(provider_id, str) or not provider_id.strip():
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action id")
                if not isinstance(process, str):
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action process date")
                if symbol is not None and not isinstance(symbol, str):
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action symbol")
                if effective is not None and not isinstance(effective, str):
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action ex-date")
                try:
                    process_date = date.fromisoformat(process)
                    effective_date = date.fromisoformat(effective) if effective else None
                    row_bytes = json.dumps(
                        row, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                except (ValueError, TypeError) as exc:
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action metadata") from exc
                item = CorporateActionObservation(
                    provider_id=provider_id,
                    kind=kind,
                    symbol=symbol,
                    effective_date=effective_date,
                    process_date=process_date,
                    observed_at=observed_at.astimezone(UTC),
                    page_hashes=(page.sha256,),
                    row_sha256=hashlib.sha256(row_bytes).hexdigest(),
                )
                previous = seen.get(provider_id)
                if previous is not None and (
                    previous.kind != item.kind or previous.row_sha256 != item.row_sha256
                ):
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", "conflicting corporate action id")
                if previous is None:
                    seen[provider_id] = item
                elif page.sha256 not in previous.page_hashes:
                    seen[provider_id] = CorporateActionObservation(
                        provider_id=previous.provider_id,
                        kind=previous.kind,
                        symbol=previous.symbol,
                        effective_date=previous.effective_date,
                        process_date=previous.process_date,
                        observed_at=previous.observed_at,
                        page_hashes=tuple(sorted((*previous.page_hashes, page.sha256))),
                        row_sha256=previous.row_sha256,
                    )
    return tuple(seen[key] for key in sorted(seen))


def make_action_capture(pages: list[RawPage], *, observed_at: datetime) -> dict[str, Any]:
    """A metadata-and-hashes record of one complete, non-executable capture."""
    if not pages:
        raise ProviderError("PROVIDER_RESPONSE_INVALID", "empty corporate action capture")
    if not isinstance(pages[0].params, dict):
        raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action page parameters")
    request = {key: value for key, value in pages[0].params.items() if key != "page_token"}
    token: str | None = None
    for index, page in enumerate(pages):
        params = page.params
        if (
            not isinstance(params, dict)
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in params.items())
            or {key: value for key, value in params.items() if key != "page_token"} != request
            or params.get("page_token") != token
        ):
            raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action pagination")
        next_token = page.payload.get("next_page_token")
        if next_token is not None and not isinstance(next_token, str):
            raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action pagination token")
        token = next_token or None
        if index < len(pages) - 1 and token is None:
            raise ProviderError("PROVIDER_RESPONSE_INVALID", "incomplete corporate action pagination")
    if token is not None:
        raise ProviderError("PROVIDER_RESPONSE_INVALID", "unfinished corporate action pagination")
    observations = observations_from_pages(pages, observed_at=observed_at)
    body = to_canonical(
        {
            "schema": CAPTURE_SCHEMA,
            "observed_at": observed_at,
            "normalization_version": NORMALIZATION_VERSION,
            "pages": [
                {"endpoint": page.endpoint, "params": page.params, "sha256": page.sha256} for page in pages
            ],
            "observations": [
                {
                    "provider_id": item.provider_id,
                    "kind": item.kind,
                    "symbol": item.symbol,
                    "effective_date": item.effective_date,
                    "process_date": item.process_date,
                    "observed_at": item.observed_at,
                    "page_hashes": item.page_hashes,
                    "row_sha256": item.row_sha256,
                    "normalization_version": item.normalization_version,
                }
                for item in observations
            ],
            "redistributable": False,
        }
    )
    body["capture_hash"] = canonical_hash(body)
    return body


def action_capture_path(library: Library, capture_hash: str) -> Path:
    if not isinstance(capture_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", capture_hash):
        raise LibraryError("DATA_MANIFEST_INVALID", "corporate action capture hash")
    return library.cache_dir / "corporate-actions" / f"{capture_hash.removeprefix('sha256:')}.json"


def store_action_capture(
    library: Library, pages: list[RawPage], *, observed_at: datetime
) -> tuple[Path, dict[str, Any]]:
    """Keep pages and manifest private in the cache; never replace a prior capture."""
    capture = make_action_capture(pages, observed_at=observed_at)
    for page in pages:
        library.store_page(page)
        library.load_page({"endpoint": page.endpoint, "params": page.params, "sha256": page.sha256})
    path = action_capture_path(library, capture["capture_hash"])
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(capture, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".capture-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise LibraryError("DATA_MANIFEST_INVALID", "corporate action capture changed") from None
    finally:
        temporary.unlink(missing_ok=True)
    return path, capture


def verify_action_capture(library: Library, path: Path) -> tuple[CorporateActionObservation, ...]:
    """Replay cached raw pages and compare every normalized observation."""
    try:
        capture = json.loads(path.read_text(encoding="utf-8"))
        capture_hash = capture.get("capture_hash") if isinstance(capture, dict) else None
        if (
            not isinstance(capture, dict)
            or capture.get("schema") != CAPTURE_SCHEMA
            or capture.get("normalization_version") != NORMALIZATION_VERSION
            or not isinstance(capture_hash, str)
            or path != action_capture_path(library, capture_hash)
            or hash_without(capture, "capture_hash") != capture_hash
        ):
            raise LibraryError("DATA_MANIFEST_INVALID", "corporate action capture identity")
        observed_at = datetime.fromisoformat(capture["observed_at"].replace("Z", "+00:00"))
        references = capture["pages"]
        if not isinstance(references, list) or not references:
            raise LibraryError("DATA_MANIFEST_INVALID", "corporate action pages")
        for ref in references:
            if (
                not isinstance(ref, dict)
                or set(ref) != {"endpoint", "params", "sha256"}
                or ref["endpoint"] != CORPORATE_ACTIONS_PATH
                or not isinstance(ref["sha256"], str)
                or not _HASH.fullmatch(ref["sha256"])
                or not isinstance(ref["params"], dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in ref["params"].items())
            ):
                raise LibraryError("DATA_MANIFEST_INVALID", "corporate action page reference")
        pages = [library.load_page(ref) for ref in references]
        rebuilt = make_action_capture(pages, observed_at=observed_at)
        if rebuilt != capture:
            raise LibraryError("DATA_MANIFEST_INVALID", "corporate action observations differ")
        return observations_from_pages(pages, observed_at=observed_at)
    except (OSError, ValueError, TypeError, KeyError, ProviderError) as exc:
        raise LibraryError("DATA_MANIFEST_INVALID", "corporate action capture unreadable") from exc
