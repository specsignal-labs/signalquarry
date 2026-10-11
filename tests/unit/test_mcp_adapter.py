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


def test_mcp_research_tools_run_variants_comparisons_and_a_study(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(mcp) as client:
            tools = {item.name for item in (await client.list_tools()).tools}
            assert {
                "sqy_diagnose",
                "sqy_runs_ls",
                "sqy_runs_show",
                "sqy_runs_compare",
                "sqy_study_init",
                "sqy_study_check",
                "sqy_study_run",
                "sqy_study_ls",
                "sqy_study_show",
            } <= tools
            project = str(tmp_path / "agent-research")
            created = await client.call_tool("sqy_init", {"path": project, "demo": True})
            assert created.structured_content["status"] == "ok"

            async def call(name: str, **arguments: Any) -> dict[str, Any]:
                result = await client.call_tool(name, {"project": project, **arguments})
                assert not result.is_error
                return result.structured_content or {}

            base = await call("sqy_backtest", strategy_id="sma-trend", label="base")
            varied = await call("sqy_backtest", strategy_id="sma-trend", params=["period=150"], label="p150")
            assert varied["data"]["params_override"] == {"period": 150}
            assert varied["data"]["configuration_hash"] != base["data"]["configuration_hash"]
            assert "benchmark_total_return" in base["metrics"]
            bad = await call("sqy_backtest", strategy_id="sma-trend", params=["period"])
            assert bad["reason_codes"] == ["USAGE_INVALID"]

            listed = await call("sqy_runs_ls", limit=5)
            assert [row["label"] for row in listed["data"]["runs"]] == ["p150", "base"]
            shown = await call("sqy_runs_show", run_id=varied["data"]["run_id"])
            assert shown["data"]["result"]["params"]["period"] == 150
            compared = await call(
                "sqy_runs_compare", run_ids=[base["data"]["run_id"], varied["data"]["run_id"]]
            )
            assert compared["data"]["comparable"] is True and compared["command"] == "runs compare"

            diagnosed = await call("sqy_diagnose", strategy_id="sma-trend", run_id=base["data"]["run_id"])
            assert diagnosed["status"] == "ok" and "regimes" in diagnosed["data"]["available"]

            scaffold = await call("sqy_study_init", strategy_id="sma-trend", study_id="first")
            assert scaffold["status"] == "ok"
            checked = await call("sqy_study_check", study_id="first")
            assert [arm["id"] for arm in checked["data"]["arms"]] == ["base", "costs-x2", "buy-and-hold"]
            ran = await call("sqy_study_run", study_id="first")
            assert ran["status"] == "ok" and ran["evidence"]["claim_level"] == "none"
            assert ran["data"]["verdict"]["outcome"] in ("supported", "not_supported", "insufficient")
            latest = await call("sqy_study_show", study_id="first")
            assert latest["data"]["study_run_id"] == ran["data"]["study_run_id"]
            studies = await call("sqy_study_ls")
            assert studies["data"]["studies"][0]["verdict"] == ran["data"]["verdict"]["outcome"]

    asyncio.run(exercise())
