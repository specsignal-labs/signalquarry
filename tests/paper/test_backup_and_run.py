# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.paper.models import PaperError
from signalquarry.api import paper as paper_api
from signalquarry.api.envelope import Envelope
from tests.paper.harness import hold_syna, rig


def test_backup_and_continuity(tmp_path: Path) -> None:
    paper = rig(tmp_path, decide=hold_syna)
    days = paper.dataset.sessions[60:66]
    paper.arm(days[0])
    for session in days[:3]:
        paper.session(session)
    outcome = paper.kernel.backup(tmp_path / "backups")
    archive = Path(outcome.data["path"])
    with tarfile.open(archive) as handle:
        assert sorted(handle.getnames()) == ["arm.json", "backup.json", "journal.jsonl"]
    for session in days[3:]:
        paper.session(session)
    assert paper.kernel.verify_continuity(archive).data["appended"] > 0

    lines = paper.deployment.journal_path.read_text().splitlines()
    paper.deployment.journal_path.write_text("\n".join(lines[:5]) + "\n")  # rolled back behind the backup
    with pytest.raises(PaperError) as info:
        paper.kernel.verify_continuity(archive)
    assert info.value.code == "PAPER_CONTINUITY_BROKEN"
    with pytest.raises(PaperError) as info:
        paper.kernel.verify_continuity(tmp_path / "missing.tar.gz")
    assert info.value.code == "PAPER_BACKUP_INVALID"


def test_paper_run_retries_until_done(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(["busy", "unavailable", "ok"])
    monkeypatch.setattr(
        paper_api,
        "paper_run_once",
        lambda alias, project=None: Envelope(command="paper run-once", status=next(replies)),
    )
    waits: list[float] = []
    envelope = api.paper_run("demo", interval=10, sleep=waits.append, clock=lambda: 0.0)
    assert envelope.status == "ok" and envelope.data["attempts"] == 3
    assert waits == [10, 10 * 2**2]  # busy retries at the interval; unavailable backs off


def test_paper_run_stops_on_blocked_and_gives_up_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        paper_api,
        "paper_run_once",
        lambda alias, project=None: Envelope(
            command="x", status="blocked", reason_codes=["PAPER_POSITION_DRIFT"]
        ),
    )
    assert api.paper_run("demo", sleep=lambda s: None, clock=lambda: 0.0).data["attempts"] == 1
    monkeypatch.setattr(
        paper_api, "paper_run_once", lambda alias, project=None: Envelope(command="x", status="busy")
    )
    now = iter(range(0, 10_000, 100))
    envelope = api.paper_run(
        "demo", interval=60, max_minutes=5, sleep=lambda s: None, clock=lambda: float(next(now))
    )
    assert envelope.status == "busy" and "PAPER_RUN_GAVE_UP" in envelope.warnings
