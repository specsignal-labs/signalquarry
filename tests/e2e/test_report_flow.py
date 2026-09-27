# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_report_renders_backtest_and_evaluation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "report-lab"
    _sqy(capsys, "init", str(project), "--demo", "--package", "report_lab_flow")
    code, payload = _sqy(capsys, "report", "--strategy", "sma-trend", "--project", str(project))
    assert (code, payload["reason_codes"]) == (65, ["REPORT_NO_RUNS"])

    assert _sqy(capsys, "backtest", "--strategy", "sma-trend", "--project", str(project))[0] == 0
    code, payload = _sqy(capsys, "evaluate", "--strategy", "sma-trend", "--project", str(project))
    assert payload["data"]["run_id"].endswith("-evaluate")
    assert any(a["kind"] == "evaluation" for a in payload["artifacts"])

    code, payload = _sqy(capsys, "report", "--strategy", "sma-trend", "--project", str(project))
    assert code == 0, payload
    kinds = {a["kind"]: a["path"] for a in payload["artifacts"]}
    assert set(kinds) == {"report", "equity"}
    text = (project / kinds["report"]).read_text()
    assert "claim level `none`" in text and "Synthetic data" in text
    assert "| G1_sample |" in text and "Hypothetical, simulated results" in text
    assert "sha256:" in text
    svg = (project / kinds["equity"]).read_text()
    assert svg.startswith("<svg") and "<polyline" in svg
    assert payload["data"]["claim_level"] == "none" and payload["evidence"]["grade"] == "synthetic"
