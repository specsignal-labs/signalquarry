# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime

import pytest

from signalquarry._internal.data.action_observations import observations_from_pages
from signalquarry._internal.data.alpaca import ProviderError, RawPage, actions_from_pages

OBSERVED = datetime(2026, 9, 28, 12, tzinfo=UTC)


def _page(actions: dict) -> RawPage:
    payload = {"corporate_actions": actions}
    body = json.dumps(payload, sort_keys=True).encode()
    return RawPage("/v1/corporate-actions", {}, body, hashlib.sha256(body).hexdigest(), payload)


def test_incomplete_unsupported_action_is_observed_but_still_blocks_dataset() -> None:
    page = _page(
        {
            "cash_mergers": [
                {"id": "event-1", "symbol": "OLD", "process_date": "2026-09-28", "cusip": "private"}
            ]
        }
    )
    (item,) = observations_from_pages([page], observed_at=OBSERVED)
    assert item.provider_id == "event-1" and item.kind == "cash_mergers"
    assert item.effective_date is None and item.process_date == date(2026, 9, 28)
    assert item.observed_at == OBSERVED and item.page_hashes == (page.sha256,)
    assert item.normalization_version == 1
    assert "private" not in repr(item)
    with pytest.raises(ProviderError, match="CORPORATE_ACTION_UNSUPPORTED"):
        actions_from_pages([page])


def test_exact_repeats_keep_all_distinct_page_provenance() -> None:
    row = {"id": "event-1", "symbol": "OLD", "process_date": "2026-09-28", "ex_date": "2026-09-29"}
    first = _page({"name_changes": [row]})
    second = _page({"name_changes": [row], "cash_mergers": []})
    (item,) = observations_from_pages([first, second, first], observed_at=OBSERVED)
    assert item.effective_date == date(2026, 9, 29)
    assert item.page_hashes == tuple(sorted((first.sha256, second.sha256)))


def test_revision_can_be_compared_between_frozen_captures() -> None:
    initial = _page({"name_changes": [{"id": "event-1", "process_date": "2026-09-28"}]})
    revised = _page(
        {"name_changes": [{"id": "event-1", "process_date": "2026-09-28", "ex_date": "2026-10-01"}]}
    )
    first = observations_from_pages([initial], observed_at=OBSERVED)[0]
    second = observations_from_pages([revised], observed_at=OBSERVED)[0]
    assert first.row_sha256 != second.row_sha256
    assert first.effective_date is None and second.effective_date == date(2026, 10, 1)


@pytest.mark.parametrize(
    "row",
    [
        {"process_date": "2026-09-28"},
        {"id": "event-1"},
        {"id": "event-1", "process_date": "invalid"},
        {"id": "event-1", "process_date": "2026-09-28", "ex_date": "invalid"},
    ],
)
def test_missing_or_invalid_revision_metadata_fails_closed(row: dict) -> None:
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        observations_from_pages([_page({"name_changes": [row]})], observed_at=OBSERVED)


def test_conflicting_id_in_one_capture_fails_closed() -> None:
    first = _page({"name_changes": [{"id": "event-1", "process_date": "2026-09-28"}]})
    second = _page({"name_changes": [{"id": "event-1", "process_date": "2026-09-29"}]})
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        observations_from_pages([first, second], observed_at=OBSERVED)


def test_page_integrity_and_observation_time_are_required() -> None:
    page = _page({"name_changes": []})
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        observations_from_pages([page], observed_at=datetime(2026, 9, 28))
    altered = RawPage(page.endpoint, page.params, page.body, "0" * 64, page.payload)
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        observations_from_pages([altered], observed_at=OBSERVED)
    altered_payload = RawPage(page.endpoint, page.params, page.body, page.sha256, {})
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        observations_from_pages([altered_payload], observed_at=OBSERVED)
