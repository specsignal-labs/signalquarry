# SPDX-License-Identifier: Apache-2.0
"""Progress: NDJSON on stderr in JSON mode, silent otherwise, never on stdout."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from signalquarry._internal.contracts import progress
from signalquarry.cli.main import main


def _events(err: str) -> list[dict]:
    return [json.loads(line) for line in err.splitlines() if line.startswith('{"')]


def test_json_mode_streams_progress_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "progress"
    main(["--json", "init", str(project), "--demo", "--package", "progress_flow"])
    capsys.readouterr()
    assert main(["--json", "check", "--project", str(project)]) == 0
    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout still holds exactly one envelope
    events = _events(captured.err)
    assert events and events[0]["type"] == "progress" and events[0]["stage"] == "check"
    assert events[0]["strategy"] == "sma-trend" and events[0]["total"] == 1

    assert main(
        ["--json", "sweep", "--strategy", "sma-trend", "--param", "period=100,150", "--project", str(project)]
    ) in (0, 2)
    stages = [e for e in _events(capsys.readouterr().err) if e["stage"] == "sweep"]
    assert [e["done"] for e in stages] == [0, 1] and stages[0]["total"] == 2


def test_emitter_is_silent_unless_configured_and_survives_closed_streams() -> None:
    progress.configure(None)
    progress.emit("noop")  # no stream: nothing happens
    buffer = io.StringIO()
    progress.configure(buffer)
    progress.emit("fetch", pages=2)
    event = json.loads(buffer.getvalue())
    assert event["stage"] == "fetch" and event["pages"] == 2 and event["elapsed_ms"] >= 0
    buffer.close()
    progress.emit("fetch", pages=3)  # a closed stream is ignored
    progress.configure(None)
