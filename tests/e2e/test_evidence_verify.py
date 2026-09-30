# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from signalquarry._internal.data.alpaca import AlpacaDataClient, ProviderError
from signalquarry._internal.paper.brokers.alpaca_paper import AlpacaPaperBroker
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.validation.ledger import ChainedLog
from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def _git(project: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=project,
        check=True,
        capture_output=True,
    )


def test_evidence_verify_detects_rewrites_against_a_base(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "ev"
    _sqy(capsys, "init", str(project), "--demo", "--package", "ev_lab")
    assert (project / ".github" / "workflows" / "check.yml").is_file()
    log = ChainedLog(project / "evidence" / "trials.jsonl", "signalquarry.trial/v1")
    log.append({"kind": "note", "value": 1})
    log.append({"kind": "note", "value": 2})
    _git(project, "init", "-q")
    _git(project, "add", "-A")
    _git(project, "commit", "-qm", "base")

    log.append({"kind": "note", "value": 3})
    code, payload = _sqy(capsys, "evidence", "verify", "--base", "HEAD", "--project", str(project))
    assert code == 0, payload
    assert payload["data"]["logs"][0]["entries"] == 3

    path = project / "evidence" / "trials.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")  # a valid chain, but shorter than the base
    assert _sqy(capsys, "evidence", "verify", "--project", str(project))[0] == 0
    code, payload = _sqy(capsys, "evidence", "verify", "--base", "HEAD", "--project", str(project))
    assert (code, payload["reason_codes"]) == (2, ["EVIDENCE_LOG_REWRITTEN"])

    path.unlink()
    code, payload = _sqy(capsys, "evidence", "verify", "--base", "HEAD", "--project", str(project))
    assert (code, payload["reason_codes"]) == (2, ["EVIDENCE_LOG_REWRITTEN"])


def test_evidence_verify_detects_a_broken_chain(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "broken"
    _sqy(capsys, "init", str(project), "--demo", "--package", "broken_lab")
    journal = project / "paper" / "demo" / "journal.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"seq":1,"prev":null,"hash":"sha256:00"}\n')
    code, payload = _sqy(capsys, "evidence", "verify", "--project", str(project))
    assert (code, payload["reason_codes"]) == (2, ["EVIDENCE_LOG_CORRUPT"])


@pytest.mark.parametrize("record", ["[]", "null", "true", '"text"', "1"])
def test_evidence_verify_rejects_non_object_json_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], record: str
) -> None:
    project = tmp_path / "non_object"
    _sqy(capsys, "init", str(project), "--demo", "--package", "non_object_lab")
    (project / "evidence" / "trials.jsonl").write_text(record + "\n")

    code, payload = _sqy(capsys, "evidence", "verify", "--project", str(project))

    assert (code, payload["reason_codes"]) == (2, ["EVIDENCE_LOG_CORRUPT"])


def test_offline_mode_forbids_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALQUARRY_OFFLINE", "1")
    with pytest.raises(ProviderError) as data_error:
        AlpacaDataClient("k", "s", max_retries=0).get_page("/v2/stocks/bars", {"symbols": "SPY"})
    assert data_error.value.code == "PROVIDER_UNAVAILABLE"
    with pytest.raises(PaperError) as broker_error:
        AlpacaPaperBroker("k", "s", max_retries=0).account()
    assert broker_error.value.code == "BROKER_UNAVAILABLE"
