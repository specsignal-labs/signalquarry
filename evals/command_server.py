# SPDX-License-Identifier: Apache-2.0
"""Trusted command proxy for agent-evaluation CLI calls."""

from __future__ import annotations

import json
import socketserver
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_REQUEST_BYTES = 65_536


def command_name(argv: list[str]) -> str:
    words = [argument for argument in argv if not argument.startswith("-")]
    command = (
        " ".join(words[:2])
        if words[:1] in (["spec"], ["paper"], ["data"], ["trials"], ["holdout"])
        else " ".join(words[:1])
    )
    return command or "unknown"


def handle_command_request(raw: bytes, invoke: Callable[[list[str]], dict[str, Any]]) -> dict[str, Any]:
    """Validate one bounded JSON request before delegating it to the evaluator."""
    if len(raw) > MAX_REQUEST_BYTES:
        return {"stdout": "", "stderr": "request too large", "exit": 64}
    try:
        request = json.loads(raw)
        argv = request["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or len(argv) > 256
            or any(not isinstance(item, str) or "\x00" in item for item in argv)
        ):
            raise ValueError("invalid argv")
        return invoke(argv)
    except (KeyError, TypeError, ValueError):
        return {"stdout": "", "stderr": "invalid command request", "exit": 64}


class _CommandHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(5)
        try:
            raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        except (OSError, TimeoutError):
            return
        response = handle_command_request(raw, self.server.invoke)  # type: ignore[attr-defined]
        try:
            self._reply(response)
        except OSError:
            return

    def _reply(self, response: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")


class EvalCommandServer(socketserver.UnixStreamServer):
    """Run requested `sqy` argv through a trusted runner and retain exit records."""

    allow_reuse_address = False
    request_queue_size = 8

    def __init__(self, socket_path: Path, runner: Callable[[list[str]], subprocess.CompletedProcess[str]]):
        self.runner = runner
        self.records: list[dict[str, Any]] = []
        super().__init__(str(socket_path), _CommandHandler)

    def invoke(self, argv: list[str]) -> dict[str, Any]:
        try:
            completed = self.runner(argv)
        except OSError as exc:
            completed = subprocess.CompletedProcess(argv, 127, "", str(exc))
        self.records.append({"command": command_name(argv), "argv": argv, "exit": completed.returncode})
        return {
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
            "exit": completed.returncode,
        }
