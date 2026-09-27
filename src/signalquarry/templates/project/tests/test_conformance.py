"""Fails if any strategy breaks the SignalQuarry contract (same checks as `sqy check`)."""

from pathlib import Path

from signalquarry.testing import conformance


def test_all_strategies_pass_conformance() -> None:
    conformance(project=Path(__file__).resolve().parents[1])
