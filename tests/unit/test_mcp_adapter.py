# SPDX-License-Identifier: Apache-2.0
"""Exercise the optional MCP adapter through the SDK's in-memory client."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from signalquarry import api

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402

from signalquarry.mcp.server import mcp  # noqa: E402


def test_mcp_discovery_and_envelope_parity(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(mcp) as client:
            tools = {item.name for item in (await client.list_tools()).tools}
            assert {"sqy_init", "sqy_check", "sqy_backtest", "sqy_evaluate", "sqy_paper_run_once"} <= tools
            assert not {"sqy_paper_arm", "sqy_trials_extend", "sqy_holdout_open"} & tools

            version = await client.call_tool("sqy_version")
            assert not version.is_error
            payload: dict[str, Any] = version.structured_content or {}
            assert payload["schema"] == "signalquarry.cli/v1"
            assert payload["command"] == "version"
            assert payload["data"] == api.version().data

            project = tmp_path / "agent-project"
            created = await client.call_tool("sqy_init", {"path": str(project), "demo": True})
            assert created.structured_content["status"] == "ok"
            listed = await client.call_tool("sqy_factor_ls", {"project": str(project)})
            assert listed.structured_content["data"]["factors"] == []
            invalid = await client.call_tool(
                "sqy_backtest", {"strategy_id": "sma-trend", "project": str(project), "start": "tomorrow"}
            )
            assert invalid.structured_content["status"] == "usage"
            assert invalid.structured_content["reason_codes"] == ["USAGE_INVALID"]

    asyncio.run(exercise())


def test_mcp_paper_runner_returns_kernel_envelope_for_missing_deployment(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(mcp) as client:
            project = tmp_path / "agent-paper"
            created = await client.call_tool("sqy_init", {"path": str(project), "demo": True})
            assert created.structured_content["status"] == "ok"
            result = await client.call_tool(
                "sqy_paper_run_once", {"alias": "missing", "project": str(project)}
            )
            assert result.structured_content["status"] != "ok"
            assert result.structured_content["schema"] == "signalquarry.cli/v1"
            assert (
                result.structured_content["reason_codes"]
                == api.paper_run_once("missing", project=project).reason_codes
            )

    asyncio.run(exercise())


def test_mcp_stdio_transport_uses_only_protocol_stdout() -> None:
    root = Path(__file__).resolve().parents[2]
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "signalquarry.mcp.server"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )

    async def exercise() -> None:
        async with Client(server) as client:
            result = await client.call_tool("sqy_version")
            assert not result.is_error
            assert result.structured_content["schema"] == "signalquarry.cli/v1"

    asyncio.run(exercise())


def test_mcp_legacy_client_can_call_same_envelope() -> None:
    async def exercise() -> None:
        async with Client(mcp, mode="legacy") as client:
            result = await client.call_tool("sqy_version")
            assert result.structured_content["schema"] == "signalquarry.cli/v1"

    asyncio.run(exercise())
