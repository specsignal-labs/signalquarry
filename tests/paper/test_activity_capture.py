# SPDX-License-Identifier: Apache-2.0
"""Read-only paper activity pagination and private capture integrity."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import signalquarry.api.paper as paper_api
from signalquarry._internal.data.alpaca import HttpResponse
from signalquarry._internal.paper.activity_capture import (
    capture_root,
    store_activity_capture,
    verify_activity_capture,
)
from signalquarry._internal.paper.activity_observations import (
    compare_activity_observations,
    observe_activity_capture,
    verify_activity_observations,
)
from signalquarry._internal.paper.brokers.alpaca_paper import PAPER_ORIGIN, AlpacaPaperBroker
from signalquarry._internal.paper.models import PaperError
from signalquarry.api.paper import (
    paper_capture_activities,
    paper_compare_observations,
    paper_observe_activities,
    paper_verify_activities,
    paper_verify_observations,
)
from signalquarry.cli.main import main

AFTER = datetime(2026, 9, 20, tzinfo=UTC)
UNTIL = datetime(2026, 9, 27, tzinfo=UTC)
OBSERVED = datetime(2026, 9, 28, tzinfo=UTC)


@dataclass
class Script:
    responses: list[object]
    urls: list[str] = field(default_factory=list)

    def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> HttpResponse:
        assert method == "GET" and url.startswith(PAPER_ORIGIN + "/v2/account")
        assert body is None and headers["APCA-API-KEY-ID"] == "key"
        self.urls.append(url)
        return HttpResponse(200, {}, json.dumps(self.responses.pop(0)).encode())


def broker(*pages: list[dict]) -> tuple[AlpacaPaperBroker, Script]:
    script = Script(list(pages))
    return AlpacaPaperBroker("key", "secret", transport=script), script


def test_complete_pagination_follows_last_id_and_preserves_raw_pages() -> None:
    first = [{"id": f"activity-{index}", "activity_type": "SPLIT"} for index in range(100)]
    second = [{"id": "activity-100", "activity_type": "MA"}]
    source, script = broker(first, second)
    pages = source.activity_pages(AFTER, UNTIL)
    assert [len(page.rows) for page in pages] == [100, 1]
    assert json.loads(pages[0].body) == first
    assert pages[0].params["after"] == AFTER.isoformat()
    assert parse_qs(urlparse(script.urls[1]).query)["page_token"] == ["activity-99"]
    assert all("category" not in page.params and "activity_types" not in page.params for page in pages)


def test_duplicate_or_truncated_pagination_is_refused() -> None:
    full = [{"id": f"activity-{index}"} for index in range(100)]
    source, _ = broker(full, [{"id": "activity-99"}])
    with pytest.raises(PaperError, match="BROKER_RESPONSE_INVALID"):
        source.activity_pages(AFTER, UNTIL)
    source, _ = broker(full)
    with pytest.raises(PaperError) as info:
        source.activity_pages(AFTER, UNTIL, max_pages=1)
    assert info.value.code == "PAPER_ACTIVITY_CAPTURE_INCOMPLETE"
    with pytest.raises(PaperError) as info:
        source.activity_pages(AFTER.replace(tzinfo=None), UNTIL)
    assert info.value.code == "USAGE_INVALID"


def test_private_capture_verifies_and_rejects_mutated_page(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    private = capture_root(tmp_path / "private", project)
    source, _ = broker([{"id": "private-activity-id", "activity_type": "REORG", "qty": "-5"}])
    pages = source.activity_pages(AFTER, UNTIL)
    account_hash = "a" * 64
    record = store_activity_capture(
        private,
        pages,
        account_sha256=account_hash,
        created_after=AFTER,
        created_until=UNTIL,
        observed_at=OBSERVED,
    )
    assert record["activities"] == 1 and record["redistributable"] is False
    assert "private-activity-id" not in json.dumps(record)
    assert verify_activity_capture(private, record["capture_hash"], account_sha256=account_hash) == record
    page_path = private / "pages" / f"{pages[0].sha256}.json"
    record_path = private / "records" / f"{record['capture_hash'].removeprefix('sha256:')}.json"
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE(page_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    page_path.chmod(0o644)
    with pytest.raises(PaperError) as info:
        verify_activity_capture(private, record["capture_hash"], account_sha256=account_hash)
    assert info.value.code == "PAPER_ACTIVITY_CACHE_UNSAFE"
    page_path.chmod(0o600)
    page_path.write_text("[]", encoding="utf-8")
    with pytest.raises(PaperError) as info:
        verify_activity_capture(private, record["capture_hash"], account_sha256=account_hash)
    assert info.value.code == "DATA_MANIFEST_INVALID"


def _stored_capture(
    root: Path,
    rows: list[dict],
    observed_at: datetime,
    *,
    created_after: datetime = AFTER,
    created_until: datetime = UNTIL,
) -> dict:
    source, _ = broker(rows)
    return store_activity_capture(
        root,
        source.activity_pages(created_after, created_until),
        account_sha256="a" * 64,
        created_after=created_after,
        created_until=created_until,
        observed_at=observed_at,
    )


def test_private_observations_replay_without_raw_terms(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    private = capture_root(tmp_path / "private", project_root)
    capture = _stored_capture(
        private,
        [{"id": "private-id", "activity_type": "DIV", "symbol": "SYN", "net_amount": "12.34"}],
        OBSERVED,
    )
    record = observe_activity_capture(private, capture["capture_hash"], account_sha256="a" * 64)
    encoded = json.dumps(record)
    assert record["observations"][0]["activity_type"] == "DIV"
    assert record["observations"][0]["fields"] == ["activity_type", "id", "net_amount", "symbol"]
    assert record["economic_terms_verified"] is False
    assert "private-id" in encoded and "12.34" not in encoded
    assert (
        verify_activity_observations(private, record["observation_hash"], account_sha256="a" * 64) == record
    )
    observation_file = private / "observations" / f"{record['observation_hash'].removeprefix('sha256:')}.json"
    assert stat.S_IMODE(observation_file.stat().st_mode) == 0o600
    with pytest.raises(PaperError) as info:
        verify_activity_observations(private, record["observation_hash"], account_sha256="b" * 64)
    assert info.value.code == "DATA_MANIFEST_INVALID"


def test_observation_comparison_reports_aggregate_revisions(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    private = capture_root(tmp_path / "private", project_root)
    left = _stored_capture(
        private,
        [
            {"id": "same", "activity_type": "DIV", "net_amount": "3"},
            {"id": "revision", "activity_type": "MA", "qty": "2"},
        ],
        OBSERVED,
    )
    right = _stored_capture(
        private,
        [
            {"id": "same", "activity_type": "DIV", "net_amount": "3"},
            {"id": "revision", "activity_type": "MA", "qty": "3"},
            {"id": "new", "activity_type": "FILL", "qty": "1"},
        ],
        OBSERVED + timedelta(hours=1),
    )
    left_observation = observe_activity_capture(private, left["capture_hash"], account_sha256="a" * 64)
    right_observation = observe_activity_capture(private, right["capture_hash"], account_sha256="a" * 64)
    comparison = compare_activity_observations(
        private,
        left_observation["observation_hash"],
        right_observation["observation_hash"],
        account_sha256="a" * 64,
    )
    assert (comparison["only_left"], comparison["only_right"]) == (0, 1)
    assert (comparison["changed"], comparison["unchanged"]) == (1, 1)
    assert "revision" not in json.dumps(comparison)
    shifted = _stored_capture(
        private,
        [{"id": "same", "activity_type": "DIV"}],
        OBSERVED + timedelta(hours=2),
        created_after=AFTER + timedelta(days=1),
        created_until=UNTIL + timedelta(days=1),
    )
    shifted_observation = observe_activity_capture(private, shifted["capture_hash"], account_sha256="a" * 64)
    with pytest.raises(PaperError) as info:
        compare_activity_observations(
            private,
            left_observation["observation_hash"],
            shifted_observation["observation_hash"],
            account_sha256="a" * 64,
        )
    assert info.value.code == "USAGE_INVALID"


def test_capture_observation_time_is_after_final_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "private"))
    started = OBSERVED + timedelta(minutes=1)
    completed = started + timedelta(minutes=2)
    times = iter((started, completed))

    class Clock:
        @staticmethod
        def now(_tz=None):
            return next(times)

    monkeypatch.setattr(paper_api, "datetime", Clock)
    source, script = broker([{"id": "activity", "activity_type": "MA"}])
    script.responses.insert(
        0,
        {"id": "synthetic-account", "status": "ACTIVE", "cash": "0", "equity": "0", "buying_power": "0"},
    )
    captured = paper_api.paper_capture_activities("demo", AFTER, UNTIL, project=root, broker=source)
    assert captured.status == "ok"
    assert captured.data["observed_at"] == completed.isoformat().replace("+00:00", "Z")


def test_capture_refuses_project_cache_unsafe_mode_and_future_window(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(PaperError) as info:
        capture_root(project / ".cache", project)
    assert info.value.code == "PAPER_ACTIVITY_CACHE_UNSAFE"
    private = capture_root(tmp_path / "private", project)
    private.chmod(0o755)
    with pytest.raises(PaperError) as info:
        capture_root(tmp_path / "private", project)
    assert info.value.code == "PAPER_ACTIVITY_CACHE_UNSAFE"
    private.chmod(0o700)
    source, _ = broker([])
    pages = source.activity_pages(AFTER, UNTIL)
    with pytest.raises(PaperError, match="BROKER_RESPONSE_INVALID"):
        store_activity_capture(
            private,
            pages,
            account_sha256="a" * 64,
            created_after=AFTER,
            created_until=OBSERVED + timedelta(days=1),
            observed_at=OBSERVED,
        )


def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "paper").mkdir(parents=True)
    (root / "signalquarry.toml").write_text("", encoding="utf-8")
    account_hash = hashlib.sha256(b"synthetic-account").hexdigest()
    (root / "paper" / "demo.paper.yaml").write_text(
        "schema: signalquarry.paper/v1\nalias: demo\nstrategy: synthetic-demo\n"
        f"broker: alpaca-paper\nexpected_account_id_sha256: {account_hash}\n",
        encoding="utf-8",
    )
    return root


def test_api_capture_and_offline_verify_never_expose_account_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "private"))
    source, script = broker([{"id": "private-activity-id", "activity_type": "SPLIT"}])
    script.responses.insert(
        0, {"id": "synthetic-account", "status": "ACTIVE", "cash": "0", "equity": "0", "buying_power": "0"}
    )
    captured = paper_capture_activities(
        "demo", AFTER, UNTIL, project=root, broker=source, observed_at=OBSERVED
    )
    assert captured.status == "ok" and captured.data["activities"] == 1
    assert "private-activity-id" not in json.dumps(captured.as_dict())
    assert "synthetic-account" not in json.dumps(captured.as_dict())
    verified = paper_verify_activities("demo", captured.data["capture_hash"], project=root)
    assert verified.status == "ok" and verified.data["activities"] == 1
    observed = paper_observe_activities("demo", captured.data["capture_hash"], project=root)
    assert observed.status == "ok" and observed.data["activities"] == 1
    verified_observation = paper_verify_observations("demo", observed.data["observation_hash"], project=root)
    assert verified_observation.status == "ok"
    compared = paper_compare_observations(
        "demo",
        observed.data["observation_hash"],
        observed.data["observation_hash"],
        project=root,
    )
    assert compared.status == "ok" and compared.data["unchanged"] == 1
    for envelope in (observed, verified_observation, compared):
        assert "private-activity-id" not in json.dumps(envelope.as_dict())
        assert "synthetic-account" not in json.dumps(envelope.as_dict())


def test_api_rejects_wrong_account_and_project_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project(tmp_path)
    source, script = broker([])
    script.responses.insert(
        0, {"id": "wrong-account", "status": "ACTIVE", "cash": "0", "equity": "0", "buying_power": "0"}
    )
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "private"))
    mismatch = paper_capture_activities(
        "demo", AFTER, UNTIL, project=root, broker=source, observed_at=OBSERVED
    )
    assert mismatch.reason_codes == ["PAPER_ACCOUNT_MISMATCH"] and not script.urls[1:]
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(root / ".cache"))
    unsafe = paper_capture_activities("demo", AFTER, UNTIL, project=root, broker=source, observed_at=OBSERVED)
    assert unsafe.reason_codes == ["PAPER_ACTIVITY_CACHE_UNSAFE"]


def test_cli_routes_creation_window_and_capture_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from signalquarry import api
    from signalquarry.api.envelope import Envelope

    seen: list[tuple] = []

    def capture(alias, after, until, *, project):
        seen.append((alias, after, until, project))
        return Envelope(command="paper capture-activities", summary="captured")

    monkeypatch.setattr(api, "paper_capture_activities", capture)
    monkeypatch.setattr(
        api,
        "paper_verify_activities",
        lambda alias, capture_hash, *, project: Envelope(
            command="paper verify-activities", summary=capture_hash
        ),
    )
    assert (
        main(
            [
                "--json",
                "paper",
                "capture-activities",
                "--alias",
                "demo",
                "--created-after",
                AFTER.isoformat(),
                "--created-until",
                UNTIL.isoformat(),
                "--project",
                str(tmp_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert seen == [("demo", AFTER, UNTIL, tmp_path)]
    assert (
        main(
            [
                "--json",
                "paper",
                "verify-activities",
                "--alias",
                "demo",
                "--capture",
                "sha256:" + "a" * 64,
                "--project",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "sha256:" in capsys.readouterr().out
    observation_hash = "sha256:" + "b" * 64
    left_hash, right_hash = "sha256:" + "c" * 64, "sha256:" + "d" * 64
    observed_calls: list[tuple] = []
    monkeypatch.setattr(
        api,
        "paper_observe_activities",
        lambda alias, capture_hash, *, project: (
            observed_calls.append(("observe", alias, capture_hash, project))
            or Envelope(command="paper observe-activities", summary=observation_hash)
        ),
    )
    monkeypatch.setattr(
        api,
        "paper_verify_observations",
        lambda alias, value, *, project: (
            observed_calls.append(("verify", alias, value, project))
            or Envelope(command="paper verify-observations", summary=value)
        ),
    )
    monkeypatch.setattr(
        api,
        "paper_compare_observations",
        lambda alias, left, right, *, project: (
            observed_calls.append(("compare", alias, left, right, project))
            or Envelope(command="paper compare-observations", summary="compared")
        ),
    )
    assert main(["--json", "paper", "observe-activities", "--alias", "demo", "--capture", left_hash]) == 0
    capsys.readouterr()
    assert (
        main(["--json", "paper", "verify-observations", "--alias", "demo", "--observation", observation_hash])
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "--json",
                "paper",
                "compare-observations",
                "--alias",
                "demo",
                "--left",
                left_hash,
                "--right",
                right_hash,
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert observed_calls == [
        ("observe", "demo", left_hash, None),
        ("verify", "demo", observation_hash, None),
        ("compare", "demo", left_hash, right_hash, None),
    ]
