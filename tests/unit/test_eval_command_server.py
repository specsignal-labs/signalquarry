# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
from typing import Any

from evals.command_server import EvalCommandServer, handle_command_request


def test_command_request_returns_and_records_the_real_cli_result() -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 2, "result\n", "gate not met\n")

    server = object.__new__(EvalCommandServer)
    server.runner = runner
    server.records = []

    raw = json.dumps({"argv": ["--json", "evaluate", "--strategy", "sma"]}).encode() + b"\n"
    response = handle_command_request(raw, server.invoke)

    assert calls == [["--json", "evaluate", "--strategy", "sma"]]
    assert response == {"stdout": "result\n", "stderr": "gate not met\n", "exit": 2}
    assert server.records == [
        {
            "command": "evaluate",
            "argv": ["--json", "evaluate", "--strategy", "sma"],
            "exit": 2,
        }
    ]


def test_command_request_rejects_invalid_input_without_recording() -> None:
    calls: list[list[str]] = []

    def invoke(argv: list[str]) -> dict[str, Any]:
        calls.append(argv)
        return {"stdout": "", "stderr": "", "exit": 0}

    response = handle_command_request(b'{"argv":["bad\\u0000arg"]}\n', invoke)

    assert response == {"stdout": "", "stderr": "invalid command request", "exit": 64}
    assert calls == []


def test_command_request_rejects_oversized_input() -> None:
    called = False

    def invoke(argv: list[str]) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"stdout": "", "stderr": "", "exit": 0}

    response = handle_command_request(b" " * 65_537, invoke)

    assert response == {"stdout": "", "stderr": "request too large", "exit": 64}
    assert called is False
