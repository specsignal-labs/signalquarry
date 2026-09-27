# SPDX-License-Identifier: Apache-2.0
"""``sqy check --parity`` reports agreement and catches divergence."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from signalquarry._internal.paper import parity as parity_module
from signalquarry._internal.paper.brokers.fake import FakeBroker
from signalquarry._internal.paper.brokers.fake_options import FakeOptionsVenue
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.paper.runner import PaperKernel
from signalquarry._internal.project.project import LoadedStrategy
from signalquarry.sdk import definition_of
from tests.options.wheel_reference import WheelParams, wheel, wheel_spec
from tests.paper.harness import Momentum, paper_spec, rotate

HERE = Path(__file__).parent


def _equity() -> LoadedStrategy:
    return LoadedStrategy(paper_spec(), definition_of(rotate), Momentum(), sys.modules[__name__], HERE)


def _options() -> LoadedStrategy:
    return LoadedStrategy(wheel_spec(), definition_of(wheel), WheelParams(), sys.modules[__name__], HERE)


def test_equity_parity_holds_with_real_rotations() -> None:
    result = parity_module.parity(_equity())
    assert result.ok, result.detail
    fills = int(result.detail.split(", ")[1].split()[0])
    assert fills >= 4


def test_equity_divergence_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    class Leaky(FakeBroker):
        def open_session(self, session):  # noqa: ANN001
            super().open_session(session)
            for order in self.orders.values():
                if order.status == "filled" and order.symbol == "SYNA":
                    self.cash -= 1  # a venue that charges something the backtest does not know about

    monkeypatch.setattr(parity_module, "FakeBroker", Leaky)
    result = parity_module.parity(_equity())
    assert not result.ok and "cash differs" in result.detail


def test_kernel_refusal_fails_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise PaperError("PAPER_DATA_STALE", "busy", "injected")

    monkeypatch.setattr(PaperKernel, "run_once", refuse)
    result = parity_module.parity(_equity())
    assert not result.ok and "PAPER_DATA_STALE" in result.detail


def test_options_parity_and_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    result = parity_module.parity(_options())
    assert result.ok, result.detail
    monkeypatch.setattr(
        parity_module,
        "FakeOptionsVenue",
        lambda data, cash: FakeOptionsVenue(data, cash, faults=["reject"] * 3),
    )
    broken = parity_module.parity(_options())
    assert not broken.ok and "wheel events differ" in broken.detail
