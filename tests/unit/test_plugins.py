# SPDX-License-Identifier: Apache-2.0
"""Plugins: discovery, add-only gates, report sections, doctor listing."""

from __future__ import annotations

import json
import sys
import types
from importlib import metadata
from pathlib import Path

import pytest

import signalquarry.plugins as plugins
from signalquarry._internal.validation.evaluate import Evaluation, claim_level
from signalquarry.cli.main import main
from signalquarry.plugins import (
    API_VERSION,
    Discovery,
    GateContext,
    GateOutcome,
    Loaded,
    ReportContext,
    discover,
)


class MinFolds:
    name = "min-folds"
    api_version = API_VERSION

    def __init__(self, minimum: int = 99) -> None:
        self.minimum = minimum

    def check(self, context: GateContext) -> GateOutcome:
        return GateOutcome(len(context.folds) >= self.minimum, {"folds": len(context.folds)})


class Broken:
    name = "broken"
    api_version = API_VERSION

    def check(self, context: GateContext) -> GateOutcome:
        raise RuntimeError("boom")


class Notes:
    name = "notes"
    api_version = API_VERSION
    title = "Desk notes"

    def render(self, context: ReportContext) -> str:
        return f"Reviewed {context.spec['id']}."


class OldVersion:
    name = "old"
    api_version = 0

    def check(self, context: GateContext) -> GateOutcome:
        return GateOutcome(True)


def _module() -> str:
    module = types.ModuleType("sq_plugin_fixture")
    module.MinFolds = MinFolds  # type: ignore[attr-defined]
    module.OldVersion = OldVersion  # type: ignore[attr-defined]
    module.Notes = Notes  # type: ignore[attr-defined]
    module.bad_name = types.SimpleNamespace(name="Bad Name", api_version=API_VERSION, check=lambda c: None)  # type: ignore[attr-defined]
    sys.modules[module.__name__] = module
    return module.__name__


def test_discovery_validates_and_reports_problems() -> None:
    name = _module()
    points = [
        metadata.EntryPoint("a", f"{name}:MinFolds", "signalquarry.gates"),
        metadata.EntryPoint("b", f"{name}:MinFolds", "signalquarry.gates"),
        metadata.EntryPoint("c", f"{name}:OldVersion", "signalquarry.gates"),
        metadata.EntryPoint("d", f"{name}:bad_name", "signalquarry.gates"),
        metadata.EntryPoint("e", "sq_plugin_missing_module:X", "signalquarry.gates"),
        metadata.EntryPoint("f", f"{name}:Notes", "signalquarry.gates"),
    ]
    found = discover("gates", entry_points=lambda group: points if group == "signalquarry.gates" else [])
    assert [p.name for p in found.plugins] == ["min-folds"]
    kinds = sorted(error.split()[0] for error in found.errors)
    assert kinds == [
        "PLUGIN_API_VERSION",
        "PLUGIN_DUPLICATE",
        "PLUGIN_INVALID",
        "PLUGIN_INVALID",
        "PLUGIN_LOAD_FAILED",
    ]
    with pytest.raises(ValueError):
        discover("live_brokers")
    assert discover("report_sections", entry_points=lambda group: []).plugins == ()


def test_plugin_gates_only_lower_claims() -> None:
    built_in = {name: {"ok": True} for name in ("G1_sample", "G2_walk_forward", "G3_stress", "G4_holdout")}
    passing = Evaluation(gates={**built_in, "plugin:x": {"ok": True}})
    failing = Evaluation(gates={**built_in, "plugin:x": {"ok": False}})
    assert claim_level(passing, grade="sip", frozen=True, conformance_ok=True) == "holdout_passed"
    assert claim_level(failing, grade="sip", frozen=True, conformance_ok=True) == "in_sample"
    unfrozen = Evaluation(gates={"plugin:x": {"ok": True}})
    assert claim_level(unfrozen, grade="sip", frozen=True, conformance_ok=True) == "in_sample"


def _fake(group_plugins: dict[str, list[object]]):
    def fake(group: str, **_: object) -> Discovery:
        items = group_plugins.get(group, [])
        loaded = tuple(Loaded(group, p.name, f"{p.name} = fixture", p) for p in items)  # type: ignore[attr-defined]
        return Discovery(loaded, ("PLUGIN_LOAD_FAILED gates:x = y: ImportError",) if group == "gates" else ())

    return fake


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_evaluate_report_and_doctor_use_plugins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import signalquarry.api

    evidence_api = sys.modules["signalquarry.api.evidence"]
    report_api = sys.modules["signalquarry.api.report"]
    assert signalquarry.api.doctor

    fake = _fake({"gates": [MinFolds(), Broken()], "report_sections": [Notes()]})
    monkeypatch.setattr(evidence_api, "discover", fake)
    monkeypatch.setattr(report_api, "discover", fake)
    monkeypatch.setattr(plugins, "discover", fake)
    project = tmp_path / "plug"
    _sqy(capsys, "init", str(project), "--demo", "--package", "plug_flow")
    code, payload = _sqy(capsys, "evaluate", "--strategy", "sma-trend", "--project", str(project))
    gates = payload["data"]["gates"]
    assert gates["plugin:min-folds"]["ok"] is False and gates["plugin:min-folds"]["folds"] >= 1
    assert gates["plugin:broken"] == {
        "ok": False,
        "error": "RuntimeError: boom",
        "source": "broken = fixture",
    }
    assert "PLUGIN_GATE_ERROR:broken" in payload["warnings"]
    assert any(w.startswith("PLUGIN_ERROR:PLUGIN_LOAD_FAILED") for w in payload["warnings"])

    code, payload = _sqy(capsys, "report", "--strategy", "sma-trend", "--project", str(project))
    assert code == 0, payload
    report = next(a for a in payload["artifacts"] if a["path"].endswith("report.md"))
    text = (project / report["path"]).read_text()
    assert "## Desk notes" in text and "Reviewed sma-trend." in text
    assert "Bootstrap 90% interval for the annual Sharpe" in text

    class Failing(Notes):
        def render(self, context: ReportContext) -> str:
            raise ValueError("no")

    monkeypatch.setattr(report_api, "discover", _fake({"report_sections": [Failing()]}))
    code, payload = _sqy(capsys, "report", "--strategy", "sma-trend", "--project", str(project))
    assert "PLUGIN_SECTION_ERROR:notes:ValueError" in payload["warnings"]

    code, payload = _sqy(capsys, "doctor")
    listed = payload["data"]["checks"]["plugins"]
    assert listed["gates"] == ["min-folds (min-folds = fixture)", "broken (broken = fixture)"]
    assert listed["ok"] is False and "PLUGIN_ERROR" in payload["warnings"]
