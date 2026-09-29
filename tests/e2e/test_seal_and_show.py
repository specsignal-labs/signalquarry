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


def test_holdout_seal_before_exploring_then_freeze_reuses_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "seal"
    _sqy(capsys, "init", str(project), "--demo", "--package", "seal_lab")
    code, payload = _sqy(capsys, "holdout", "seal", "--strategy", "sma-trend", "--project", str(project))
    assert code == 0 and payload["data"]["newly_sealed"] is True
    assert ledger.latest_freeze(project, "sma-trend") is None
    start = payload["data"]["holdout_start"]
    code, payload = _sqy(capsys, "holdout", "seal", "--strategy", "sma-trend", "--project", str(project))
    assert payload["data"] == {"family": "sma-trend", "holdout_start": start, "newly_sealed": False}
    code, payload = _sqy(capsys, "backtest", "--strategy", "sma-trend", "--project", str(project))
    assert "HOLDOUT_CLIPPED" in payload["warnings"]
    code, payload = _sqy(capsys, "spec", "freeze", "--strategy", "sma-trend", "--project", str(project))
    assert payload["data"]["holdout_start"] == start and payload["data"]["newly_sealed"] is False
    assert ledger.latest_freeze(project, "sma-trend")["freeze_hash"] == payload["data"]["freeze_hash"]


def test_trials_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "show"
    _sqy(capsys, "init", str(project), "--demo", "--package", "show_lab")
    code, payload = _sqy(capsys, "trials", "show", "abc", "--project", str(project))
    assert (code, payload["reason_codes"]) == (65, ["TRIAL_NOT_FOUND"])
    ledger.record_trial(
        project,
        {
            "kind": "trial",
            "family": "f",
            "configuration_hash": "sha256:" + "ab" * 32,
            "dataset_identity": "d",
        },
    )
    code, payload = _sqy(capsys, "trials", "show", "abab", "--project", str(project))
    assert code == 0 and payload["data"]["entries"][0]["seq"] == 1
    assert _sqy(capsys, "trials", "show", "1", "--project", str(project))[0] == 0
