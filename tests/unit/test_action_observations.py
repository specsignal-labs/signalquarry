# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.action_observations import (
    action_capture_path,
    make_action_capture,
    observations_from_pages,
    store_action_capture,
    verify_action_capture,
)
from signalquarry._internal.data.alpaca import AlpacaDataClient, ProviderError, RawPage, actions_from_pages
from signalquarry._internal.data.library import Library, LibraryError
from signalquarry.api.data import data_capture_actions, data_ls, data_verify
from signalquarry.api.envelope import Envelope
from signalquarry.cli.main import main

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


def test_capture_is_private_immutable_and_replays_pages(tmp_path: Path) -> None:
    row = {
        "id": "event-1",
        "symbol": "OLD",
        "process_date": "2026-09-28",
        "ex_date": "2026-09-29",
        "cusip": "private-raw-term",
        "cash_amount": "123.45",
    }
    page = _page({"cash_mergers": [row]})
    library = Library(tmp_path / "cache", tmp_path / "project" / "data" / "manifests")
    path, capture = store_action_capture(library, [page], observed_at=OBSERVED)
    assert path.parent == library.cache_dir / "corporate-actions"
    assert path.read_text(encoding="utf-8").find("private-raw-term") == -1
    assert "123.45" not in path.read_text(encoding="utf-8")
    assert capture["redistributable"] is False
    assert verify_action_capture(library, path) == observations_from_pages([page], observed_at=OBSERVED)
    assert store_action_capture(library, [page], observed_at=OBSERVED)[0] == path
    later = datetime(2026, 9, 29, 12, tzinfo=UTC)
    later_path, later_capture = store_action_capture(library, [page], observed_at=later)
    assert later_path != path and later_capture["capture_hash"] != capture["capture_hash"]
    assert path.is_file()


def test_capture_tampering_and_missing_pages_fail_closed(tmp_path: Path) -> None:
    page = _page({"name_changes": [{"id": "event-1", "process_date": "2026-09-28"}]})
    library = Library(tmp_path / "cache", tmp_path / "manifests")
    path, _ = store_action_capture(library, [page], observed_at=OBSERVED)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["observations"][0]["kind"] = "cash_mergers"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        verify_action_capture(library, path)
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        store_action_capture(library, [page], observed_at=OBSERVED)
    path.unlink()
    store_action_capture(library, [page], observed_at=OBSERVED)
    library.page_path(page.sha256).unlink()
    with pytest.raises(LibraryError, match="DATA_PAGE_MISSING"):
        verify_action_capture(library, path)


def test_empty_capture_is_invalid() -> None:
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        make_action_capture([], observed_at=OBSERVED)


def test_capture_requires_complete_pagination() -> None:
    payload = {"corporate_actions": {"name_changes": []}, "next_page_token": "next"}
    body = json.dumps(payload).encode()
    first = RawPage("/v1/corporate-actions", {}, body, hashlib.sha256(body).hexdigest(), payload)
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        make_action_capture([first], observed_at=OBSERVED)
    last = _page({"name_changes": []})
    final = RawPage(last.endpoint, {"page_token": "next"}, last.body, last.sha256, last.payload)
    assert make_action_capture([first, final], observed_at=OBSERVED)["observations"] == []
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        make_action_capture([first, last], observed_at=OBSERVED)
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        make_action_capture(
            [RawPage(first.endpoint, {"page_token": "next"}, first.body, first.sha256, first.payload)],
            observed_at=OBSERVED,
        )
    numeric_token = {"corporate_actions": {"name_changes": []}, "next_page_token": 123}
    numeric_body = json.dumps(numeric_token).encode()
    malformed = RawPage(
        first.endpoint, {}, numeric_body, hashlib.sha256(numeric_body).hexdigest(), numeric_token
    )
    with pytest.raises(ProviderError, match="PROVIDER_RESPONSE_INVALID"):
        make_action_capture([malformed], observed_at=OBSERVED)


def test_verified_capture_rejects_self_consistent_metadata_edits(tmp_path: Path) -> None:
    page = _page({"name_changes": [{"id": "event-1", "process_date": "2026-09-28"}]})
    library = Library(tmp_path / "cache", tmp_path / "manifests")
    _, original = store_action_capture(library, [page], observed_at=OBSERVED)

    def forged(changes: dict) -> Path:
        body = {key: value for key, value in original.items() if key != "capture_hash"}
        body.update(changes)
        capture_hash = canonical_hash(body)
        path = action_capture_path(library, capture_hash)
        path.write_text(json.dumps({**body, "capture_hash": capture_hash}), encoding="utf-8")
        return path

    altered = [dict(original["observations"][0], kind="cash_mergers")]
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        verify_action_capture(library, forged({"observations": altered}))
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        verify_action_capture(library, forged({"pages": []}))
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        verify_action_capture(
            library, forged({"pages": [{"endpoint": "/bad", "params": {}, "sha256": page.sha256}]})
        )
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        action_capture_path(library, "../bad")


def test_capture_api_records_unsupported_action_without_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "signalquarry.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    page = _page({"cash_mergers": [{"id": "event-1", "process_date": "2026-09-28"}]})

    class Client:
        def corporate_actions(self, symbols: tuple[str, ...], start: date, end: date) -> list[RawPage]:
            assert symbols == ("OLD",) and start == end == date(2026, 9, 28)
            return [page]

    result = data_capture_actions(
        ("OLD",),
        date(2026, 9, 28),
        date(2026, 9, 28),
        project=root,
        client=cast(AlpacaDataClient, Client()),
        observed_at=OBSERVED,
    )
    assert result.status == "ok" and result.data["observations"] == 1
    assert result.data["cache_record"].startswith("corporate-actions/")
    assert not (root / "data").exists()
    verified = data_verify(project=root)
    assert verified.status == "ok"
    assert verified.data["datasets"] == [
        {"action_capture": result.data["cache_record"], "observations": 1, "ok": True}
    ]
    assert data_ls(project=root).data["action_captures"] == [
        {"capture_hash": result.data["capture_hash"], "cache_record": result.data["cache_record"]}
    ]
    assert data_capture_actions((), date(2026, 9, 28), date(2026, 9, 28), project=root).status == "usage"

    def unwritable(*args: object, **kwargs: object) -> None:
        raise OSError("private cache path")

    monkeypatch.setattr("signalquarry.api.data.store_action_capture", unwritable)
    unavailable = data_capture_actions(
        ("OLD",),
        date(2026, 9, 28),
        date(2026, 9, 28),
        project=root,
        client=cast(AlpacaDataClient, Client()),
        observed_at=OBSERVED,
    )
    assert unavailable.status == "unavailable"
    assert unavailable.reason_codes == ["CACHE_DIR_NOT_WRITABLE"]


def test_capture_cli_routes_explicit_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    received: list[object] = []

    def capture(symbols: tuple[str, ...], start: date, end: date, *, project: Path | None) -> Envelope:
        received.extend((symbols, start, end, project))
        return Envelope(command="data capture-actions", summary="recorded")

    monkeypatch.setattr("signalquarry.cli.main.api.data_capture_actions", capture)
    assert (
        main(
            [
                "--json",
                "data",
                "capture-actions",
                "--symbols",
                "spy,qqq",
                "--start",
                "2026-09-01",
                "--end",
                "2026-09-30",
                "--project",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert received == [
        ("SPY", "QQQ"),
        date(2026, 9, 1),
        date(2026, 9, 30),
        tmp_path,
    ]
    assert json.loads(capsys.readouterr().out)["command"] == "data capture-actions"
