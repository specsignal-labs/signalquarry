# SPDX-License-Identifier: Apache-2.0
"""``sqy backtest --param NAME=VALUE --label TEXT``: one-off variants of a strategy."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from signalquarry._internal.validation import ledger
from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def _lab(tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str) -> Path:
    project = tmp_path / name
    _sqy(capsys, "init", str(project), "--demo", "--package", name.replace("-", "_"))
    return project


def _result(project: Path, payload: dict) -> dict:
    path = next(a["path"] for a in payload["artifacts"] if a["kind"] == "result")
    return json.loads((project / path).read_text())


def test_an_override_is_its_own_configuration_and_leaves_the_yaml_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _lab(tmp_path, capsys, "override-lab")
    spec_path = next(project.glob("src/*/sma_trend/strategy.yaml"))
    before = spec_path.read_text()
    run = ("backtest", "--strategy", "sma-trend", "--project", str(project))

    code, base = _sqy(capsys, *run)
    code, varied = _sqy(capsys, *run, "--param", "period=150", "--label", "p150")
    assert code == 0, varied
    assert spec_path.read_text() == before
    assert varied["data"]["configuration_hash"] != base["data"]["configuration_hash"]
    assert varied["data"]["ledger_hash"] != base["data"]["ledger_hash"]
    assert varied["data"]["dataset_identity"] == base["data"]["dataset_identity"]
    assert varied["data"]["params_override"] == {"period": 150} and varied["data"]["label"] == "p150"
    assert "params_override" not in base["data"] and "label" not in base["data"]
    assert varied["next_actions"][0]["command"].startswith("edit strategy.yaml params")
    assert base["next_actions"][0]["command"] == "sqy spec freeze --strategy sma-trend"

    document = _result(project, varied)
    assert document["params"] == {"symbol": "SYNA", "period": 150, "weight": "0.95"}
    assert document["label"] == "p150" and document["command"] == "backtest"
    assert _result(project, base)["params"]["period"] == 200

    # The same variant through a sweep is the same configuration and the same run.
    code, swept = _sqy(
        capsys, "sweep", "--strategy", "sma-trend", "--param", "period=150", "--project", str(project)
    )
    (point,) = swept["data"]["points"]
    assert point["configuration_hash"] == varied["data"]["configuration_hash"]
    point_result = json.loads(
        (project / ".signalquarry" / "runs" / point["run_id"] / "result.json").read_text()
    )
    assert point_result["ledger_hash"] == varied["data"]["ledger_hash"]

    # Several overrides, and a plain run afterwards is the base configuration again.
    code, two = _sqy(capsys, *run, "--param", "period=150", "--param", "weight=0.5")
    assert _result(project, two)["params"] == {"symbol": "SYNA", "period": 150, "weight": "0.5"}
    code, again = _sqy(capsys, *run)
    assert again["data"]["configuration_hash"] == base["data"]["configuration_hash"]
    assert again["data"]["ledger_hash"] == base["data"]["ledger_hash"]


@pytest.mark.parametrize(
    ("extra", "code", "reason"),
    [
        (("--param", "period"), 64, "USAGE_INVALID"),
        (("--param", "=150"), 64, "USAGE_INVALID"),
        (("--param", "period="), 64, "USAGE_INVALID"),
        (("--param", "period=150", "--param", "period=100"), 64, "USAGE_INVALID"),
        (("--label", "has space"), 64, "USAGE_INVALID"),
        (("--label", "-leading"), 64, "USAGE_INVALID"),
        (("--label", "x" * 65), 64, "USAGE_INVALID"),
        (("--param", "period=5"), 65, "STRATEGY_PARAMS_INVALID"),
        (("--param", "nope=1"), 65, "STRATEGY_PARAMS_INVALID"),
    ],
)
def test_bad_overrides_and_labels_are_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: tuple[str, ...], code: int, reason: str
) -> None:
    project = _lab(tmp_path, capsys, "override-bad")
    got, payload = _sqy(capsys, "backtest", "--strategy", "sma-trend", "--project", str(project), *extra)
    assert (got, payload["reason_codes"]) == (code, [reason]), payload
    assert not (project / ".signalquarry" / "runs").exists()


def test_a_label_may_use_the_whole_allowed_alphabet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _lab(tmp_path, capsys, "override-label")
    label = "A1." + "b_c-" * 15 + "z"
    assert len(label) == 64
    code, payload = _sqy(
        capsys, "backtest", "--strategy", "sma-trend", "--label", label, "--project", str(project)
    )
    assert code == 0 and payload["data"]["label"] == label
    assert _result(project, payload)["label"] == label


def test_on_real_data_an_override_is_a_trial_within_the_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _lab(tmp_path, capsys, "override-real")
    spec_path = next(project.glob("src/*/sma_trend/strategy.yaml"))
    spec_path.write_text(spec_path.read_text().replace("trial_budget: 50", "trial_budget: 2"))
    resolve_api = sys.modules["signalquarry.api.resolve"]
    real = resolve_api.resolve

    def historical(command, strategy_id, project):
        resolved = real(command, strategy_id, project)
        return replace(resolved, grade="historical", dataset_id="test-dataset")

    monkeypatch.setattr(resolve_api, "resolve", historical)
    run = ("backtest", "--strategy", "sma-trend", "--project", str(project))

    code, first = _sqy(capsys, *run, "--param", "period=150")
    assert code == 0 and first["evidence"]["trial"]["count"] == 1
    code, second = _sqy(capsys, *run, "--param", "period=160")
    assert code == 0 and second["evidence"]["trial"]["count"] == 2
    runs = sorted((project / ".signalquarry" / "runs").iterdir())

    code, refused = _sqy(capsys, *run, "--param", "period=170")
    assert (code, refused["reason_codes"]) == (2, ["TRIAL_BUDGET_EXHAUSTED"])
    assert "1 new trials would exceed family sma-trend's budget (2/2 used)" in refused["summary"]
    assert sorted((project / ".signalquarry" / "runs").iterdir()) == runs
    assert len(ledger.trials(project).entries()) == 2

    # A configuration that was already tried costs nothing and still runs.
    code, repeat = _sqy(capsys, *run, "--param", "period=150")
    assert code == 0 and repeat["data"]["ledger_hash"] == first["data"]["ledger_hash"]
    assert len(ledger.trials(project).entries()) == 2
