"""The drill family's strategies meet the SignalQuarry contract (the checks `sqy check` runs)."""

from pathlib import Path

from signalquarry.testing import conformance

LAB = Path(__file__).resolve().parents[3]


def test_drill_conformance() -> None:
    conformance(project=LAB, strategy="drill")
