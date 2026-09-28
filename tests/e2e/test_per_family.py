# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalquarry._internal.validation import ledger
from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def _trial(family: str, configuration: str) -> dict:
    return {
        "kind": "trial",
        "family": family,
        "configuration_hash": "sha256:" + configuration * 64,
        "dataset_identity": "d",
        "sharpe": 0.1,
    }


@pytest.fixture
def lab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    project = tmp_path / "family-lab"
    _sqy(capsys, "init", str(project), "--demo", "--package", "family_lab")
    config = project / "signalquarry.toml"
    config.write_text(config.read_text() + '\n[evidence]\nlayout = "per_family"\n')
    return project


def test_family_logs_index_and_project_wide_count(lab: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger.record_trial(lab, _trial("alpha", "a"))
    ledger.record_trial(lab, _trial("alpha", "b"))
    ledger.record_trial(lab, _trial("beta", "c"))
    assert ledger.record_trial(lab, _trial("beta", "c"))[1] is False  # same configuration and dataset
    assert (lab / "families/alpha/evidence/trials.jsonl").is_file()
    assert (lab / "families/beta/evidence/trials.jsonl").is_file()
    assert not (lab / "evidence/trials.jsonl").exists()
    summary = ledger.trial_summary(lab, "beta")
    assert (summary["project_count"], summary["family_count"]) == (3, 1)
    assert summary["head"] == ledger.project_index(lab).entries()[-1]["hash"]
    assert (lab / "evidence/project_index.jsonl").is_file()
    assert ledger.project_index(lab).path == lab / "evidence/project_index.jsonl"
    assert ledger.project_index(lab).entries()[0]["schema"] == "signalquarry.project-index/v1"

    assert _sqy(capsys, "spec", "freeze", "--strategy", "sma-trend", "--project", str(lab))[0] == 0
    assert (lab / "families/sma-trend/evidence/freezes.jsonl").is_file()
    code, payload = _sqy(capsys, "trials", "ls", "--project", str(lab))
    assert code == 0 and len(payload["data"]["entries"]) == 3
    code, payload = _sqy(capsys, "evidence", "verify", "--project", str(lab))
    assert code == 0, payload
    assert {row["path"] for row in payload["data"]["logs"]} >= {
        "families/alpha/evidence/trials.jsonl",
        "families/beta/evidence/trials.jsonl",
        "families/sma-trend/evidence/freezes.jsonl",
        "evidence/project_index.jsonl",
    }


def test_rolled_back_family_log_is_detected(lab: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger.record_trial(lab, _trial("alpha", "a"))
    ledger.record_trial(lab, _trial("alpha", "b"))
    path = lab / "families/alpha/evidence/trials.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")  # still a valid chain, one entry shorter
    assert ledger.verify_index(lab) == ["EVIDENCE_INDEX_MISMATCH:alpha/trials"]
    code, payload = _sqy(capsys, "evidence", "verify", "--project", str(lab))
    assert (code, payload["reason_codes"]) == (2, ["EVIDENCE_INDEX_MISMATCH"])
    path.unlink()
    assert ledger.verify_index(lab) == ["EVIDENCE_INDEX_MISMATCH:alpha/trials"]


def test_family_log_without_project_index_is_detected(lab: Path) -> None:
    assert ledger.record_trial(lab, _trial("alpha", "a"))[1] is True
    (lab / "evidence/project_index.jsonl").unlink()
    assert ledger.verify_index(lab) == ["EVIDENCE_INDEX_MISMATCH:alpha/trials"]


def test_invalid_layout_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "signalquarry.toml").write_text('[evidence]\nlayout = "flat"\n')
    with pytest.raises(ledger.LedgerError) as info:
        ledger.layout(tmp_path)
    assert info.value.code == "PROJECT_CONFIG_INVALID"
    (tmp_path / "signalquarry.toml").write_text('[evidence]\nlayout = "per_family"\n')
    with pytest.raises(ledger.LedgerError) as info:
        ledger.trials(tmp_path)
    assert info.value.code == "EVIDENCE_FAMILY_REQUIRED"
