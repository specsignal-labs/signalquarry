# SPDX-License-Identifier: Apache-2.0
"""Documented activity labels remain private observations, never resolved terms."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from signalquarry._internal.paper.activity_capture import (
    _write_immutable,
    capture_root,
    store_activity_capture,
)
from signalquarry._internal.paper.activity_decoder import classify_activity_row
from signalquarry._internal.paper.activity_observations import (
    OBSERVATION_SCHEMA_V1,
    _make_v1_activity_observations,
    make_activity_observations,
    observe_activity_capture,
    verify_activity_observations,
)
from signalquarry._internal.paper.brokers.alpaca_paper import BrokerActivityPage
from signalquarry._internal.paper.models import PaperError

AFTER = datetime(2026, 9, 20, tzinfo=UTC)
UNTIL = datetime(2026, 9, 27, tzinfo=UTC)
OBSERVED = datetime(2026, 9, 28, tzinfo=UTC)
ACCOUNT_HASH = "a" * 64


@pytest.mark.parametrize(
    ("activity_type", "subtype", "family"),
    [
        ("SPLIT", "FSPLIT", "split"),
        ("SPLIT", "RSPLIT", "split"),
        ("SPLIT", "USPLIT", "split"),
        ("MA", "CMA", "cash_merger_activity"),
        ("MA", "SMA", "stock_merger_activity"),
        ("MA", "SCMA", "stock_cash_merger_activity"),
        ("NC", "SNC", "symbol_name_change"),
        ("NC", "CNC", "cusip_name_change"),
        ("NC", "SCNC", "symbol_and_cusip_name_change"),
        ("REORG", "WRM", "worthless_removal"),
    ],
)
def test_documented_pairs_are_classified_without_execution_authority(
    activity_type: str, subtype: str, family: str
) -> None:
    result = classify_activity_row(
        {"id": "synthetic-activity", "activity_type": activity_type, "activity_sub_type": subtype}
    )
    assert result.documented_pair
    assert result.documented_family == family
    assert result.activity_sub_type == subtype
    assert result.correction_present is False
    assert result.paper_eligible is False


@pytest.mark.parametrize(
    "row",
    [
        {"id": "synthetic-activity", "activity_type": "DIV", "activity_sub_type": "CDIV"},
        {"id": "synthetic-activity", "activity_type": "MA", "activity_sub_type": "NEW"},
        {"id": "synthetic-activity", "activity_type": "SPLIT"},
    ],
)
def test_unmapped_activity_pairs_remain_unclassified(row: dict[str, object]) -> None:
    result = classify_activity_row(row)
    assert result.documented_pair is False
    assert result.documented_family is None
    assert result.paper_eligible is False


def test_correction_reference_is_only_flagged_and_never_retained() -> None:
    result = classify_activity_row(
        {
            "id": "synthetic-correction",
            "activity_type": "MA",
            "activity_sub_type": "CMA",
            "previous_id": "synthetic-prior-activity",
        }
    )
    assert result.documented_pair
    assert result.correction_present
    assert result.paper_eligible is False
    assert "synthetic-prior-activity" not in repr(result)


@pytest.mark.parametrize(
    "row",
    [
        {"activity_type": "MA", "activity_sub_type": "CMA"},
        {"id": " ", "activity_type": "MA", "activity_sub_type": "CMA"},
        {"id": "synthetic-activity", "activity_type": " "},
        {"id": "synthetic-activity", "activity_type": "MA", "activity_sub_type": 1},
        {"id": "synthetic-activity", "activity_type": "MA", "activity_subtype": "CMA"},
        {
            "id": "synthetic-activity",
            "activity_type": "MA",
            "activity_sub_type": "CMA",
            "previous_id": 1,
        },
    ],
)
def test_malformed_or_wrong_wire_shapes_fail_closed(row: dict[str, object]) -> None:
    with pytest.raises(PaperError) as info:
        classify_activity_row(row)
    assert info.value.code == "BROKER_RESPONSE_INVALID"


def _fixture_page(rows: list[dict[str, object]]) -> BrokerActivityPage:
    body = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return BrokerActivityPage(
        params={
            "after": AFTER.isoformat(),
            "until": UNTIL.isoformat(),
            "direction": "asc",
            "page_size": "100",
        },
        body=body,
        sha256=hashlib.sha256(body).hexdigest(),
        rows=tuple(rows),
    )


def _private_root(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    return capture_root(tmp_path / "private", project)


def test_v2_observation_records_category_flags_but_not_activity_amounts(tmp_path: Path) -> None:
    private = _private_root(tmp_path)
    rows = [
        {
            "id": "synthetic-activity",
            "activity_type": "MA",
            "activity_sub_type": "SCMA",
            "symbol": "SYN",
            "qty": "7",
            "net_amount": "123.45",
        }
    ]
    capture = store_activity_capture(
        private,
        [_fixture_page(rows)],
        account_sha256=ACCOUNT_HASH,
        created_after=AFTER,
        created_until=UNTIL,
        observed_at=OBSERVED,
    )

    record = observe_activity_capture(private, capture["capture_hash"], account_sha256=ACCOUNT_HASH)
    item = record["observations"][0]
    encoded = json.dumps(record)
    assert record["schema"] == "signalquarry.paper-activity-observations/v2"
    assert item["activity_sub_type"] == "SCMA"
    assert item["documented_family"] == "stock_cash_merger_activity"
    assert item["documented_pair"] is True
    assert item["paper_eligible"] is False
    assert record["event_links_verified"] is False
    assert record["economic_terms_verified"] is False
    assert "123.45" not in encoded and '"qty": "7"' not in encoded
    assert (
        verify_activity_observations(private, record["observation_hash"], account_sha256=ACCOUNT_HASH)
        == record
    )


def test_v1_observation_records_remain_verifiable(tmp_path: Path) -> None:
    private = _private_root(tmp_path)
    page = _fixture_page([{"id": "synthetic-activity", "activity_type": "DIV"}])
    capture = store_activity_capture(
        private,
        [page],
        account_sha256=ACCOUNT_HASH,
        created_after=AFTER,
        created_until=UNTIL,
        observed_at=OBSERVED,
    )
    legacy = _make_v1_activity_observations(capture, [page])
    assert legacy["schema"] == OBSERVATION_SCHEMA_V1
    digest = legacy["observation_hash"].removeprefix("sha256:")
    _write_immutable(
        private / "observations" / f"{digest}.json",
        (json.dumps(legacy, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )
    assert (
        verify_activity_observations(private, legacy["observation_hash"], account_sha256=ACCOUNT_HASH)
        == legacy
    )


def test_new_observation_builder_requires_a_documented_rest_envelope() -> None:
    page = _fixture_page([{"id": "synthetic-activity", "activity_type": "MA", "activity_subtype": "CMA"}])
    capture = {
        "capture_hash": "sha256:" + "0" * 64,
        "account_sha256": ACCOUNT_HASH,
        "created_after": AFTER,
        "created_until": UNTIL,
        "observed_at": OBSERVED,
    }
    with pytest.raises(PaperError) as info:
        make_activity_observations(capture, [page])
    assert info.value.code == "BROKER_RESPONSE_INVALID"
