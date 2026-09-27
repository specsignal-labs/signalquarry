# SPDX-License-Identifier: Apache-2.0
"""`broker: plugin:<name>`: paper brokers from the signalquarry.brokers entry point, paper_only enforced."""

from __future__ import annotations

import json
import uuid
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

import signalquarry.plugins as plugins
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.paper.brokers.fake import FakeBroker
from signalquarry.cli.main import main
from signalquarry.plugins import API_VERSION, Discovery, Loaded


class PaperVenue(FakeBroker):
    """A complete paper broker (the framework's fake) with a calendar."""

    def __init__(self) -> None:
        super().__init__(
            synthetic_dataset(date(2024, 1, 2), date(2024, 3, 1), symbols=("SYNA",)), Decimal(100000)
        )

    def calendar(self, start, end):  # noqa: ANN001
        return [s for s in self.dataset.sessions if start <= s <= end]


class HalfVenue:
    paper_only = True
    name = "half-x"

    def calendar(self, start, end):  # noqa: ANN001
        return []


class LiveVenue:
    paper_only = False
    name = "live-x"


class Plugin:
    api_version = API_VERSION

    def __init__(self, name: str, broker: type) -> None:
        self.name, self.broker, self.calls = name, broker, []

    def create(self, alias: str, key_id: str | None, secret_key: str | None):  # noqa: ANN201
        self.calls.append((alias, key_id, secret_key))
        return self.broker()


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def _project(tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str = "equity") -> Path:
    project = tmp_path / f"brokers-{kind}"
    _sqy(capsys, "init", str(project), "--demo", "--kind", kind, "--package", "b" + uuid.uuid4().hex[:10])
    return project


def _use(project: Path, broker: str) -> None:
    config = project / "paper" / "demo.paper.yaml"
    text = config.read_text()
    assert "broker: simulated" in text
    config.write_text(text.replace("broker: simulated", f"broker: {broker}"))


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> dict[str, Plugin]:
    found = {
        "venue-x": Plugin("venue-x", PaperVenue),
        "live-x": Plugin("live-x", LiveVenue),
        "half-x": Plugin("half-x", HalfVenue),
    }

    def fake(group: str, **_: object) -> Discovery:
        items = found.values() if group == "brokers" else []
        return Discovery(tuple(Loaded(group, p.name, f"{p.name} = fixture", p) for p in items))

    monkeypatch.setattr(plugins, "discover", fake)
    return found


def test_plugin_brokers_are_found_and_paper_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], installed: dict[str, Plugin]
) -> None:
    project = _project(tmp_path, capsys)
    _use(project, "plugin:missing")
    code, payload = _sqy(capsys, "paper", "dry-run", "--alias", "demo", "--project", str(project))
    assert payload["reason_codes"] == ["PAPER_BROKER_PLUGIN_NOT_FOUND"]
    _use_again = project / "paper" / "demo.paper.yaml"
    _use_again.write_text(_use_again.read_text().replace("plugin:missing", "plugin:live-x"))
    code, payload = _sqy(capsys, "paper", "dry-run", "--alias", "demo", "--project", str(project))
    assert payload["reason_codes"] == ["BROKER_NOT_PAPER_ONLY"] and code == 2
    _use_again.write_text(_use_again.read_text().replace("plugin:live-x", "plugin:half-x"))
    code, payload = _sqy(capsys, "paper", "dry-run", "--alias", "demo", "--project", str(project))
    assert payload["reason_codes"] == ["BROKER_PLUGIN_INVALID"] and "submit" in payload["summary"]
    _use_again.write_text(_use_again.read_text().replace("plugin:half-x", "plugin:live-x"))
    _use_again.write_text(_use_again.read_text().replace("plugin:live-x", "plugin:venue-x"))
    code, payload = _sqy(capsys, "paper", "dry-run", "--alias", "demo", "--project", str(project))
    # The plugin broker was built; the synthetic demo strategy is then refused on a real paper venue.
    assert installed["venue-x"].calls and installed["venue-x"].calls[0][0] == "demo"
    assert payload["reason_codes"] == ["PAPER_CONFIG_INVALID"], payload


def test_options_and_invalid_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], installed: dict[str, Plugin]
) -> None:
    options = _project(tmp_path, capsys, "options")
    _use(options, "plugin:venue-x")
    code, payload = _sqy(capsys, "paper", "dry-run", "--alias", "demo", "--project", str(options))
    assert payload["reason_codes"] == ["PAPER_KIND_UNSUPPORTED"]
    equity = _project(tmp_path, capsys)
    _use(equity, "plugin:Bad Name")
    code, payload = _sqy(capsys, "paper", "dry-run", "--alias", "demo", "--project", str(equity))
    assert code == 65
    code, payload = _sqy(capsys, "doctor")
    assert payload["data"]["checks"]["plugins"]["brokers"][0] == "venue-x (venue-x = fixture)"
