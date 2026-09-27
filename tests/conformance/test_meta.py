# SPDX-License-Identifier: Apache-2.0
"""Meta-tests: the conformance checks must catch deliberately broken strategies and engines."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.engine import backtest as engine
from signalquarry._internal.project.project import load_config, load_strategies
from signalquarry._internal.validation.conformance import run_checks
from signalquarry.testing import conformance


@pytest.fixture
def project(tmp_path: Path) -> Path:
    # A unique package per test: strategy modules are cached in sys.modules by name.
    package = "meta_" + uuid.uuid4().hex[:10]
    root = tmp_path / "meta"
    assert api.init(root, demo=True, package=package).status == "ok"
    return root


def _package(root: Path) -> str:
    return next(p.name for p in (root / "src").iterdir() if p.is_dir())


def _strategy_file(root: Path) -> Path:
    return root / "src" / _package(root) / "sma_trend" / "strategy.py"


def _checks(root: Path) -> dict[str, bool]:
    strategy = load_strategies(load_config(root))["sma-trend"]
    return {item.name: item.ok for item in run_checks(strategy)}


def test_clean_demo_passes(project: Path) -> None:
    assert all(_checks(project).values())
    assert conformance(project)[0]["ok"] is True


def test_off_by_one_context_builder_is_caught(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = engine._Views.bars

    def leaky(self, symbol: str, start: int, stop: int):  # includes the bar of the session being traded
        return original(self, symbol, start + 1, min(stop + 1, len(self.dataset.sessions)))

    monkeypatch.setattr(engine._Views, "bars", leaky)
    assert _checks(project)["lookahead"] is False


def test_module_state_that_changes_decisions_is_caught(project: Path) -> None:
    path = _strategy_file(project)
    path.write_text(
        path.read_text()
        .replace(
            "def decide(ctx: Ctx, p: SmaParams) -> Decision:",
            'SEEN: list[int] = []\n\n\n@strategy(params=SmaParams, lookback=lambda p: p.period)\ndef decide(ctx: Ctx, p: SmaParams) -> Decision:\n    SEEN.append(1)\n    if len(SEEN) % 7 == 0:\n        return Decision.target({}, "PRICE_BELOW_SMA")',
            1,
        )
        .replace("@strategy(params=SmaParams, lookback=lambda p: p.period)\nSEEN", "SEEN", 1)
    )
    checks = _checks(project)
    assert checks["determinism"] is False or checks["lookahead"] is False
    with pytest.raises(AssertionError):
        conformance(project)


@pytest.mark.parametrize(
    "line",
    ["import os", "import random", "from datetime import datetime", "import urllib.request", "import time"],
)
def test_forbidden_imports_are_caught(project: Path, line: str) -> None:
    path = _strategy_file(project)
    path.write_text(line + "\n" + path.read_text())
    assert _checks(project)["import_policy"] is False
