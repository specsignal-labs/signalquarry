# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from signalquarry.cli.main import main

ENVELOPE_KEYS = {
    "schema",
    "command",
    "framework_version",
    "run_id",
    "duration_ms",
    "status",
    "reason_codes",
    "summary",
    "data",
    "metrics",
    "evidence",
    "artifacts",
    "warnings",
    "next_actions",
}


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_version_envelope(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(capsys, "version")
    assert code == 0
    assert set(payload) == ENVELOPE_KEYS
    assert payload["schema"] == "signalquarry.cli/v1"
    assert payload["data"]["canonical"] == "v2"


def test_doctor_reports_missing_credentials_as_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    code, payload = _run(capsys, "doctor")
    assert code == 0
    assert "DATA_CREDENTIALS_MISSING" in payload["warnings"]
    assert payload["data"]["checks"]["dependencies"]["ok"] is True


def test_schema_list_get_and_unknown(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(capsys, "schema")
    assert code == 0 and "evidence-bundle" in payload["data"]["schemas"]
    code, payload = _run(capsys, "schema", "evidence-bundle")
    assert code == 0 and payload["data"]["schema"]["title"] == "EvidenceBundleV1"
    code, payload = _run(capsys, "schema", "nope")
    assert code == 65 and payload["reason_codes"] == ["SCHEMA_UNKNOWN"]


def test_explain_and_commands(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(capsys, "explain", "holdout_reused")
    assert code == 0 and payload["data"]["category"] == "evidence"
    code, payload = _run(capsys, "explain", "NOT_A_CODE")
    assert code == 65
    code, payload = _run(capsys, "commands")
    assert {item["name"] for item in payload["data"]["commands"]} >= {
        "version",
        "doctor",
        "schema",
        "explain",
        "commands",
    }
    assert payload["data"]["exit_codes"]["usage"] == 64


def test_usage_error_exits_64(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(capsys, "no-such-command")
    assert code == 64
    assert payload["status"] == "usage" and payload["reason_codes"] == ["USAGE_INVALID"]


def test_unexpected_errors_still_print_one_envelope(monkeypatch, capsys) -> None:
    import json

    import pytest

    from signalquarry import api
    from signalquarry.cli.main import main

    def explode(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(api, "version", explode)
    assert main(["--json", "version"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error" and payload["reason_codes"] == ["INTERNAL_ERROR"]
    trace = payload["data"]["trace_id"]
    assert f"trace {trace}" in captured.err and "RuntimeError: kaboom" in captured.err
    monkeypatch.setenv("SIGNALQUARRY_DEBUG", "1")
    with pytest.raises(RuntimeError):
        main(["--json", "version"])


def test_large_data_moves_to_an_artifact(monkeypatch, capsys, tmp_path) -> None:
    import json

    from signalquarry import api
    from signalquarry.api.envelope import Envelope
    from signalquarry.cli.main import main

    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path))
    big = {"rows": ["x" * 100] * 1000}
    monkeypatch.setattr(api, "version", lambda: Envelope(command="version", data=big))
    assert main(["--json", "version"]) == 0
    payload = json.loads(capsys.readouterr().out)
    moved = payload["data"]["moved_to_artifact"]
    assert "DATA_MOVED_TO_ARTIFACT" in payload["warnings"] and payload["data"]["keys"] == ["rows"]
    assert json.loads(open(moved).read()) == big
    assert payload["artifacts"][-1]["kind"] == "data" and payload["artifacts"][-1]["sha256"]

    for flags in (["--detail", "full"], ["--detail=full"]):
        assert main(["--json", *flags, "version"]) == 0
        assert json.loads(capsys.readouterr().out)["data"] == big
    assert main(["--json", "version", "--detail", "loud"]) == 64
    assert json.loads(capsys.readouterr().out)["reason_codes"] == ["USAGE_INVALID"]
    assert main(["--json", "--detail=summary", "version"]) == 0
    assert "moved_to_artifact" in json.loads(capsys.readouterr().out)["data"]


def test_duration_covers_the_whole_command(monkeypatch, capsys) -> None:
    import json
    import time

    from signalquarry import api
    from signalquarry.api.envelope import Envelope
    from signalquarry.cli.main import main

    def slow() -> Envelope:
        time.sleep(0.05)
        return Envelope(command="version")  # built at the end, as most handlers do

    monkeypatch.setattr(api, "version", slow)
    assert main(["--json", "version"]) == 0
    assert json.loads(capsys.readouterr().out)["duration_ms"] >= 50


def test_doctor_reports_the_ots_client(monkeypatch, capsys, tmp_path) -> None:
    import json
    import os

    from signalquarry.cli.main import main

    monkeypatch.setenv("PATH", str(tmp_path))
    main(["--json", "doctor"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["checks"]["opentimestamps"] == {"ok": True, "client": None, "installed": False}
    assert "OTS_CLIENT_MISSING" in payload["warnings"]
    fake = tmp_path / "ots"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    main(["--json", "doctor"])
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["data"]["checks"]["opentimestamps"]["installed"] is True
        and os.fspath(fake) == payload["data"]["checks"]["opentimestamps"]["client"]
    )
