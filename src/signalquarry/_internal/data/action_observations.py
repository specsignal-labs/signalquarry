# SPDX-License-Identifier: Apache-2.0
"""Versioned, non-executable observations of Alpaca corporate-action rows.

These records deliberately contain no consideration or successor mapping. They
can be inspected and compared across captures, but cannot drive a ledger or
authorize paper orders. Raw pages remain the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime

from signalquarry._internal.data.alpaca import CORPORATE_ACTIONS_PATH, ProviderError, RawPage

NORMALIZATION_VERSION = 1
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
