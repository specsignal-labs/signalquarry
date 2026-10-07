# SPDX-License-Identifier: Apache-2.0
"""Arm token, journal and run lease: what each records, refuses and how it fails."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from signalquarry._internal.canonical import canonical_hash, hash_without
from signalquarry._internal.paper import arm
from signalquarry._internal.paper.journal import SCHEMA, Journal
from signalquarry._internal.paper.lease import RunLease
from signalquarry._internal.paper.models import PaperError

NOW = datetime(2026, 9, 29, 13, 0, tzinfo=UTC)
ACCOUNT = "account-123"


def _token(**changes: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "alias": "demo",
        "strategy_id": "momentum",
        "configuration_hash": "sha256:" + "a" * 64,
        "freeze_hash": "sha256:" + "b" * 64,
        "account_id": ACCOUNT,
        "broker": "alpaca-paper",
        "journal_head": None,
        "now": NOW,
        "days": 30,
    }
    options.update(changes)
    return arm.issue(**options)


def _armed_journal(tmp_path: Path, token: dict[str, Any] | None = None) -> tuple[Journal, dict[str, Any]]:
    token = token or _token()
    journal = Journal.open(tmp_path / "journal.jsonl")
    journal.append("armed", {"token_hash": token["token_hash"]}, now=NOW)
    return journal, token


def _check(token: dict[str, Any] | None, journal: Journal, **changes: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "alias": "demo",
        "configuration_hash": "sha256:" + "a" * 64,
        "freeze_hash": "sha256:" + "b" * 64,
        "account_id": ACCOUNT,
        "now": NOW + timedelta(days=1),
    }
    options.update(changes)
    return arm.check(token, journal, **options)


def _refusal(token: dict[str, Any] | None, journal: Journal, **changes: Any) -> PaperError:
    with pytest.raises(PaperError) as info:
        _check(token, journal, **changes)
    return info.value


def test_an_issued_token_binds_every_input_and_hashes_the_account() -> None:
    token = _token(journal_head="sha256:" + "c" * 64)
    assert token["schema"] == "signalquarry.arm-token/v1"
    assert (
        token["alias"] == "demo" and token["strategy_id"] == "momentum" and token["broker"] == "alpaca-paper"
    )
    assert (
        token["configuration_hash"] == "sha256:" + "a" * 64 and token["freeze_hash"] == "sha256:" + "b" * 64
    )
    assert token["journal_head"] == "sha256:" + "c" * 64
    assert (
        token["account_id_sha256"] == hashlib.sha256(ACCOUNT.encode()).hexdigest() == arm.sha256_hex(ACCOUNT)
    )
    assert ACCOUNT not in json.dumps(token)
    assert token["issued_at"] == "2026-09-29T13:00:00Z" and token["expires_at"] == "2026-10-29T13:00:00Z"
    assert token["token_hash"] == hash_without(token, "token_hash")
    assert _token()["token_hash"] == _token()["token_hash"]
    for change in (
        {"alias": "other"},
        {"days": 31},
        {"account_id": "x"},
        {"journal_head": "sha256:" + "d" * 64},
    ):
        assert _token(**change)["token_hash"] != _token()["token_hash"]


def test_times_are_recorded_in_utc_whatever_zone_they_were_given_in() -> None:
    token = _token(now=datetime(2026, 9, 29, 9, 0, tzinfo=timezone(timedelta(hours=-4))), days=1)
    assert (token["issued_at"], token["expires_at"]) == ("2026-09-29T13:00:00Z", "2026-09-30T13:00:00Z")


def test_the_token_file_round_trips_sorted_and_unreadable_files_read_as_none(tmp_path: Path) -> None:
    token = _token()
    path = tmp_path / "deep" / "arm.json"
    arm.write(path, token)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n") and text.startswith('{\n  "account_id_sha256"')
    assert arm.read(path) == token
    assert arm.read(tmp_path / "missing.json") is None
    for content in ("{broken", "[]", "5", '"x"'):
        path.write_text(content, encoding="utf-8")
        assert arm.read(path) is None
    assert arm.read(tmp_path) is None  # a directory is not a token file


def test_a_matching_token_passes_and_is_returned(tmp_path: Path) -> None:
    journal, token = _armed_journal(tmp_path)
    assert _check(token, journal) is token


def test_no_token_or_no_arm_entry_means_not_armed(tmp_path: Path) -> None:
    journal, token = _armed_journal(tmp_path)
    error = _refusal(None, journal)
    assert (error.code, error.status, error.detail) == ("PAPER_NOT_ARMED", "disabled", "")
    empty = Journal.open(tmp_path / "other.jsonl")
    error = _refusal(token, empty)
    assert (error.code, error.status, error.detail) == ("PAPER_NOT_ARMED", "disabled", "")
    journal.append("disarmed", {}, now=NOW)
    assert _refusal(token, journal).code == "PAPER_NOT_ARMED"


def test_a_halt_after_arming_reports_its_reason_codes(tmp_path: Path) -> None:
    journal, token = _armed_journal(tmp_path)
    journal.append("halted", {"reason_codes": ["A_ONE", "B_TWO"]}, now=NOW)
    error = _refusal(token, journal)
    assert (error.code, error.status, error.detail) == ("PAPER_HALTED", "disabled", "A_ONE,B_TWO")
    journal.append("resumed", {}, now=NOW)
    assert _refusal(token, journal).code == "PAPER_HALTED"  # resuming does not re-arm
    bare, bare_token = _armed_journal(tmp_path / "second")
    bare.append("halted", {}, now=NOW)
    assert _refusal(bare_token, bare).detail == ""
    journal.append("armed", {"token_hash": token["token_hash"]}, now=NOW)
    assert _check(token, journal) is token


def test_a_token_that_does_not_match_the_journal_entry_is_refused(tmp_path: Path) -> None:
    journal, token = _armed_journal(tmp_path)
    detail = "arm token does not match the journal"
    tampered = {**token, "freeze_hash": "sha256:" + "e" * 64}
    wrong_schema = {**token, "schema": "signalquarry.arm-token/v2"}
    other = _token(alias="demo", days=29)
    for candidate in (tampered, wrong_schema, other):
        error = _refusal(candidate, journal)
        assert (error.code, error.status, error.detail) == ("PAPER_NOT_ARMED", "disabled", detail)
    no_hash = {key: value for key, value in token.items() if key != "token_hash"}
    assert _refusal(no_hash, journal).detail == detail


def test_the_alias_must_match(tmp_path: Path) -> None:
    journal, token = _armed_journal(tmp_path)
    error = _refusal(token, journal, alias="other")
    assert (error.code, error.status, error.detail) == ("PAPER_NOT_ARMED", "disabled", "alias")


def test_expiry_is_exclusive_at_the_expiry_instant(tmp_path: Path) -> None:
    journal, token = _armed_journal(tmp_path)
    expires = NOW + timedelta(days=30)
    assert _check(token, journal, now=expires - timedelta(seconds=1)) is token
    error = _refusal(token, journal, now=expires)
    assert (error.code, error.status, error.detail) == (
        "PAPER_ARM_EXPIRED",
        "disabled",
        "2026-10-29T13:00:00Z",
    )
    assert _refusal(token, journal, now=expires + timedelta(days=1)).code == "PAPER_ARM_EXPIRED"


def test_a_changed_configuration_or_freeze_makes_the_arm_stale(tmp_path: Path) -> None:
    journal, token = _armed_journal(tmp_path)
    for change in (
        {"configuration_hash": "sha256:" + "0" * 64},
        {"freeze_hash": "sha256:" + "0" * 64},
        {"freeze_hash": None},
    ):
        error = _refusal(token, journal, **change)
        assert (error.code, error.status) == ("PAPER_ARM_STALE", "disabled")
        assert error.detail == "strategy configuration or freeze changed"


def test_a_different_account_is_blocked_not_merely_disabled(tmp_path: Path) -> None:
    journal, token = _armed_journal(tmp_path)
    error = _refusal(token, journal, account_id="another-account")
    assert (error.code, error.status) == ("PAPER_ACCOUNT_MISMATCH", "blocked")
    assert error.detail == "broker account differs from the armed account"


def test_checks_run_in_a_fixed_order_so_the_most_basic_reason_wins(tmp_path: Path) -> None:
    journal, token = _armed_journal(tmp_path)
    everything_wrong = {
        "alias": "other",
        "now": NOW + timedelta(days=99),
        "configuration_hash": "x",
        "account_id": "y",
    }
    assert _refusal(token, journal, **everything_wrong).detail == "alias"
    del everything_wrong["alias"]
    assert _refusal(token, journal, **everything_wrong).code == "PAPER_ARM_EXPIRED"
    del everything_wrong["now"]
    assert _refusal(token, journal, **everything_wrong).code == "PAPER_ARM_STALE"


def test_the_journal_chains_entries_in_canonical_json_and_assigns_sequence_numbers(tmp_path: Path) -> None:
    journal = Journal.open(tmp_path / "sub" / "journal.jsonl")
    assert journal.head is None and journal.entries == []
    first = journal.append("armed", {"token_hash": "t"}, now=NOW)
    second = journal.append("session_started", {"session": "2026-09-29"}, now=NOW + timedelta(minutes=1))
    assert (first["seq"], first["prev"]) == (1, None)
    assert (second["seq"], second["prev"]) == (2, first["hash"])
    assert journal.head == second["hash"]
    assert first["schema"] == SCHEMA == "signalquarry.paper-journal/v1"
    assert (first["kind"], first["at"], first["token_hash"]) == ("armed", "2026-09-29T13:00:00Z", "t")
    body = {key: value for key, value in second.items() if key != "hash"}
    assert second["hash"] == canonical_hash(body)
    lines = (tmp_path / "sub" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first, second]
    reopened = Journal.open(tmp_path / "sub" / "journal.jsonl")
    assert reopened.entries == journal.entries and reopened.head == journal.head


def test_appending_is_durable_and_the_default_time_is_utc_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda descriptor: (synced.append(descriptor), real(descriptor))[1])
    journal = Journal.open(tmp_path / "journal.jsonl")
    before = datetime.now(UTC)
    entry = journal.append("armed", {})
    assert len(synced) == 1
    assert (
        before - timedelta(seconds=2)
        <= datetime.fromisoformat(entry["at"].replace("Z", "+00:00"))
        <= datetime.now(UTC)
    )
    local = journal.append("halted", {}, now=datetime(2026, 1, 1, 9, 0, tzinfo=timezone(timedelta(hours=-5))))
    assert local["at"] == "2026-01-01T14:00:00Z"


def test_reserved_fields_cannot_be_overwritten_by_a_body(tmp_path: Path) -> None:
    journal = Journal.open(tmp_path / "journal.jsonl")
    with pytest.raises(ValueError, match="^JOURNAL_FIELD_RESERVED:hash,kind,seq$"):
        journal.append("armed", {"seq": 9, "hash": "x", "kind": "y", "ok": 1})
    for name in ("schema", "at", "prev"):
        with pytest.raises(ValueError, match=f"^JOURNAL_FIELD_RESERVED:{name}$"):
            journal.append("armed", {name: 1})
    assert journal.entries == [] and not (tmp_path / "journal.jsonl").exists()


def test_of_kind_and_last_select_by_kind_in_order(tmp_path: Path) -> None:
    journal = Journal.open(tmp_path / "journal.jsonl")
    for kind in ("armed", "session_started", "session_completed", "session_started", "halted"):
        journal.append(kind, {}, now=NOW)
    assert [entry["seq"] for entry in journal.of_kind("session_started")] == [2, 4]
    assert [entry["seq"] for entry in journal.of_kind("armed", "halted")] == [1, 5]
    assert journal.of_kind("resumed") == []
    assert journal.last("session_started")["seq"] == 4  # type: ignore[index]
    assert journal.last("armed", "session_completed")["seq"] == 3  # type: ignore[index]
    assert journal.last("resumed") is None


def test_a_tampered_or_foreign_journal_is_blocked_on_open(tmp_path: Path) -> None:
    journal = Journal.open(tmp_path / "journal.jsonl")
    journal.append("armed", {"token_hash": "t"}, now=NOW)
    journal.append("halted", {}, now=NOW)
    path = tmp_path / "journal.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["token_hash"] = "forged"
    path.write_text(json.dumps(first) + "\n" + lines[1] + "\n", encoding="utf-8")
    with pytest.raises(PaperError) as info:
        Journal.open(path)
    assert (info.value.code, info.value.status) == ("PAPER_JOURNAL_CORRUPT", "blocked")
    assert info.value.detail


def test_the_lease_is_exclusive_records_the_pid_and_releases_on_exit(tmp_path: Path) -> None:
    path = tmp_path / "state" / "run.lock"
    with RunLease(path) as lease:
        assert lease.path == path and path.read_text() == str(os.getpid())
        assert path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(PaperError) as info, RunLease(path):
            pass
        assert (info.value.code, info.value.status, info.value.detail) == (
            "RUN_LEASE_BUSY",
            "busy",
            str(path),
        )
        assert path.read_text() == str(os.getpid())  # the loser did not touch the holder's record
    with RunLease(path):
        pass


def test_a_released_lease_can_be_retaken_and_shorter_pids_leave_no_stale_digits(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    path.write_text("9999999999999")
    with RunLease(path):
        assert path.read_text() == str(os.getpid())
    lease = RunLease(path)
    lease.__exit__(None, None, None)  # exiting a lease that was never entered is harmless
    with lease:
        pass
    with lease:
        pass


def test_the_lease_is_released_when_the_body_raises(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with pytest.raises(RuntimeError, match="boom"), RunLease(path):
        raise RuntimeError("boom")
    with RunLease(path):
        pass
