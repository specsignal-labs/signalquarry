# SPDX-License-Identifier: Apache-2.0
"""Documentation-backed labels for read-only Alpaca account activities.

This module classifies only the provider's activity-type/subtype pair. It does
not parse economic terms, resolve an asset, or produce a lifecycle event.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from signalquarry._internal.paper.models import PaperError

_DOCUMENTED_FAMILIES = {
    ("SPLIT", "FSPLIT"): "split",
    ("SPLIT", "RSPLIT"): "split",
    ("SPLIT", "USPLIT"): "split",
    ("MA", "CMA"): "cash_merger_activity",
    ("MA", "SMA"): "stock_merger_activity",
    ("MA", "SCMA"): "stock_cash_merger_activity",
    ("NC", "SNC"): "symbol_name_change",
    ("NC", "CNC"): "cusip_name_change",
    ("NC", "SCNC"): "symbol_and_cusip_name_change",
    ("REORG", "WRM"): "worthless_removal",
}


@dataclass(frozen=True)
class ActivityClassification:
    """A provider category label with explicit execution and evidence limits."""

    activity_type: str
    activity_sub_type: str | None
    documented_family: str | None
    correction_present: bool

    @property
    def documented_pair(self) -> bool:
        return self.documented_family is not None

    @property
    def paper_eligible(self) -> bool:
        return False


def classify_activity_row(row: Mapping[str, Any]) -> ActivityClassification:
    """Classify one legacy REST row without interpreting its economic fields.

    The adapter reads ``/v2/account/activities``. Alpaca documents that
    non-trade REST activity types are retained in Activity SSE and maps REST
    ``activity_sub_type`` to SSE ``activity_subtype``. This allowlist uses
    those documented labels only; a real private cassette is still required
    to validate the provider's exact activity terms and account effects.
    """
    activity_id = row.get("id")
    activity_type = row.get("activity_type")
    subtype = row.get("activity_sub_type")
    if not isinstance(activity_id, str) or not activity_id.strip():
        raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity id")
    if not isinstance(activity_type, str) or not activity_type or activity_type != activity_type.strip():
        raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity type")
    if "activity_subtype" in row:
        raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity REST subtype field")
    if subtype is not None and (not isinstance(subtype, str) or not subtype or subtype != subtype.strip()):
        raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity subtype")

    previous_id = row.get("previous_id")
    if previous_id is not None and (not isinstance(previous_id, str) or not previous_id.strip()):
        raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity correction reference")

    return ActivityClassification(
        activity_type=activity_type,
        activity_sub_type=subtype,
        documented_family=_DOCUMENTED_FAMILIES.get((activity_type, subtype)) if subtype else None,
        correction_present=previous_id is not None,
    )
