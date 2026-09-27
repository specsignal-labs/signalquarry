# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path

import pytest

from signalquarry._internal.paper.journal import Journal
from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def lab(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SIGNALQUARRY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + "/usr/bin:/bin")  # no ots client
    project = tmp_path / "commit-lab"
    # A unique package per test: strategy modules are cached in sys.modules by name.
    _sqy(capsys, "init", str(project), "--demo", "--package", "commit_" + uuid.uuid4().hex[:10])
    return project


def test_spec_commitment_reveal_and_verify(
    lab: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _sqy(capsys, "commit", "create", "--strategy", "sma-trend", "--project", str(lab))
    assert (code, payload["reason_codes"]) == (2, ["FREEZE_REQUIRED"])
    assert _sqy(capsys, "spec", "freeze", "--strategy", "sma-trend", "--project", str(lab))[0] == 0
    code, payload = _sqy(capsys, "commit", "create", "--strategy", "sma-trend", "--project", str(lab))
    assert code == 0, payload
    assert payload["data"]["proof"] == "pending" and "COMMITMENT_PROOF_PENDING" in payload["warnings"]
    record_path = lab / payload["data"]["path"]
    record = json.loads(record_path.read_text())
    assert set(record) >= {"c_spec", "c_code", "digest"} and "salt" not in json.dumps(record)
    salt_file = tmp_path / "config" / "salts" / "commit-lab" / f"{record['id']}.json"
    assert stat.S_IMODE(salt_file.stat().st_mode) == 0o600
    assert not list(lab.rglob(f"{record['id']}.json")) or all(
        "salt" not in p.read_text() for p in lab.rglob("*.json")
    )

    opening = tmp_path / "opening.json"
    assert (
        _sqy(capsys, "commit", "reveal", "--id", record["id"], "--out", str(opening), "--project", str(lab))[
            0
        ]
        == 0
    )
    code, payload = _sqy(capsys, "commit", "verify", "--record", str(record_path), "--reveal", str(opening))
    assert code == 0, payload
    tampered = json.loads(opening.read_text())
    tampered["params"]["period"] = 150
    opening.write_text(json.dumps(tampered))
    code, payload = _sqy(capsys, "commit", "verify", "--record", str(record_path), "--reveal", str(opening))
    assert (code, payload["reason_codes"]) == (2, ["COMMITMENT_SPEC_MISMATCH"])
    tampered = json.loads(salt_file.read_text())
    tampered["code_tree_hash"] = "sha256:" + "0" * 64
    opening.write_text(json.dumps(tampered))
    code, payload = _sqy(capsys, "commit", "verify", "--record", str(record_path), "--reveal", str(opening))
    assert "COMMITMENT_CODE_MISMATCH" in payload["reason_codes"]


def test_stale_freeze_and_missing_salt(lab: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _sqy(capsys, "spec", "freeze", "--strategy", "sma-trend", "--project", str(lab))
    strategy = next(lab.glob("src/*/sma_trend/strategy.py"))
    strategy.write_text(strategy.read_text() + "\n# changed after the freeze\n")
    code, payload = _sqy(capsys, "commit", "create", "--strategy", "sma-trend", "--project", str(lab))
    assert payload["reason_codes"] == ["FREEZE_STALE"]
    code, payload = _sqy(
        capsys, "commit", "reveal", "--id", "nope", "--out", str(lab / "x.json"), "--project", str(lab)
    )
    assert (code, payload["reason_codes"]) == (69, ["COMMITMENT_SALT_MISSING"])


def test_journal_commitment_and_opentimestamps(
    lab: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _sqy(
        capsys, "commit", "create", "--strategy", "sma-trend", "--alias", "demo", "--project", str(lab)
    )
    assert payload["reason_codes"] == ["PAPER_CONFIG_NOT_FOUND"]
    journal = Journal.open(lab / "paper" / "demo" / "journal.jsonl")
    journal.append("note", {"body": "x"})
    fake = tmp_path / "bin" / "ots"
    fake.parent.mkdir(exist_ok=True)
    fake.write_text('#!/bin/sh\n[ "$1" = stamp ] && printf proof > "$2.ots"\n')
    fake.chmod(0o755)
    code, payload = _sqy(
        capsys, "commit", "create", "--strategy", "sma-trend", "--alias", "demo", "--project", str(lab)
    )
    assert code == 0 and payload["data"]["proof"] == "submitted", payload
    record = json.loads((lab / payload["data"]["path"]).read_text())
    assert record["journal_head"] == journal.head and (lab / (payload["data"]["path"] + ".ots")).is_file()
