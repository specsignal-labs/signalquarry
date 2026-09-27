# SPDX-License-Identifier: Apache-2.0
"""Credential values never appear in reprs, envelopes or tracebacks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.data.alpaca import AlpacaDataClient
from signalquarry._internal.data.credentials import load_data_credentials, redact, secret_values
from signalquarry.api.envelope import Envelope
from signalquarry.cli.main import main

KEY, SECRET = "PKTESTKEY12345", "sEcReT-value-0123456789"


def test_reprs_hide_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert SECRET not in repr(AlpacaDataClient(KEY, SECRET)) and KEY not in repr(
        AlpacaDataClient(KEY, SECRET)
    )
    monkeypatch.setenv("SIGNALQUARRY_CONFIG_DIR", str(tmp_path))
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "credentials.toml"
    path.write_text('[data]\nkey_id = "FILEKEY-abcdef"\nsecret_key = "FILESECRET-abcdef"\n')
    os.chmod(path, 0o600)
    loaded = load_data_credentials()
    assert loaded is not None and "FILESECRET" not in repr(loaded) and "credentials.toml" in repr(loaded)
    assert {"FILEKEY-abcdef", "FILESECRET-abcdef"} <= secret_values(environ={})  # file values count too
    assert redact("x FILESECRET-abcdef y") == "x [REDACTED] y"


def test_cli_output_and_traces_are_scrubbed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", SECRET)
    monkeypatch.setattr(
        api, "version", lambda: Envelope(command="version", summary=f"leaked {SECRET}", data={"k": KEY})
    )
    assert main(["--json", "version"]) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and KEY not in out and json.loads(out)["data"] == {"k": "[REDACTED]"}

    def explode() -> Envelope:
        raise RuntimeError(f"auth failed for {KEY}/{SECRET}")

    monkeypatch.setattr(api, "version", explode)
    assert main(["--json", "version"]) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err and KEY not in captured.out + captured.err
    assert "[REDACTED]" in captured.err and "[REDACTED]" in json.loads(captured.out)["summary"]
