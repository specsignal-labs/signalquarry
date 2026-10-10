# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.stats import pbo_cscv
from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_pbo_detects_noise_and_real_edges() -> None:
    rng = np.random.Generator(np.random.PCG64(1))
    noise = rng.normal(0, 0.01, (1000, 20))
    assert 0.25 < pbo_cscv(noise)["pbo"] < 0.75
    edge = noise.copy()
    edge[:, 0] += 0.004  # one configuration is genuinely better in every period
    assert pbo_cscv(edge)["pbo"] < 0.05
    assert np.isnan(pbo_cscv(noise[:, :1])["pbo"]) and pbo_cscv(noise[:10])["splits"] == 0


@pytest.fixture
def lab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    project = tmp_path / "sweep-lab"
    _sqy(capsys, "init", str(project), "--demo", "--package", "sweep_" + uuid.uuid4().hex[:8])
    return project


def test_sweep_on_synthetic_data(lab: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _sqy(
        capsys, "sweep", "--strategy", "sma-trend", "--param", "period=50,100,150,200", "--project", str(lab)
    )
    assert code == 0, payload
    assert len(payload["data"]["points"]) == 4 and payload["data"]["pbo"]["splits"] == 252
    assert len({p["configuration_hash"] for p in payload["data"]["points"]}) == 4
    assert (
        payload["evidence"]["claim_level"] == "none" and "SWEEP_RESULTS_ARE_IN_SAMPLE" in payload["warnings"]
    )
    assert not (lab / "evidence" / "trials.jsonl").exists()  # synthetic runs are not trials
    code, payload = _sqy(
        capsys, "sweep", "--strategy", "sma-trend", "--param", "period=5", "--project", str(lab)
    )
    assert payload["reason_codes"] == ["STRATEGY_PARAMS_INVALID"]
    code, payload = _sqy(
        capsys, "sweep", "--strategy", "sma-trend", "--param", "period", "--project", str(lab)
    )
    assert code == 64
    code, payload = _sqy(
        capsys,
        "sweep",
        "--strategy",
        "sma-trend",
        "--param",
        "period=" + ",".join(str(20 + k) for k in range(201)),
        "--project",
        str(lab),
    )
    assert payload["reason_codes"] == ["SWEEP_TOO_LARGE"]


def test_sweep_on_real_data_records_trials_and_respects_the_budget(
    lab: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep_api = sys.modules[
        "signalquarry.api.sweep"
    ]  # the package re-exports the function under the same name
    real = sweep_api.resolve

    def historical(command, strategy_id, project):
        resolved = real(command, strategy_id, project)
        return replace(resolved, grade="historical", dataset_id="test-dataset")

    monkeypatch.setattr(sweep_api, "resolve", historical)
    code, payload = _sqy(
        capsys, "sweep", "--strategy", "sma-trend", "--param", "period=50,100", "--project", str(lab)
    )
    assert code == 0 and payload["evidence"]["trial"]["project_count"] == 2
    assert len(ledger.trials(lab).entries()) == 2
    spec = next(lab.glob("src/*/sma_trend/strategy.yaml"))
    spec.write_text(spec.read_text().replace("trial_budget: 50", "trial_budget: 3"))
    code, payload = _sqy(
        capsys, "sweep", "--strategy", "sma-trend", "--param", "period=60,70", "--project", str(lab)
    )
    assert (code, payload["reason_codes"]) == (2, ["TRIAL_BUDGET_EXHAUSTED"])


def test_sweep_persists_every_point_as_a_run_and_records_the_sweep(
    lab: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _sqy(
        capsys, "sweep", "--strategy", "sma-trend", "--param", "period=150,200", "--project", str(lab)
    )
    assert code == 0, payload
    points = payload["data"]["points"]
    assert [p["params"] for p in points] == [{"period": 150}, {"period": 200}]
    for point in points:
        run_dir = lab / ".signalquarry" / "runs" / point["run_id"]
        assert {p.name for p in run_dir.iterdir()} == {
            "result.json",
            "equity.csv",
            "benchmark.csv",
            "fills.csv",
            "decisions.jsonl",
        }
        result = json.loads((run_dir / "result.json").read_text())
        assert result["command"] == "sweep" and result["configuration_hash"] == point["configuration_hash"]
        assert result["params"]["period"] == point["params"]["period"]
        assert result["spec"]["data"]["symbols"] == ["SYNA"] and "hypothesis" not in result["spec"]
        assert result["metrics"]["sharpe"] == point["sharpe"]
        assert result["metrics"]["fills"] == point["fills"] > 0
        assert float(result["metrics"]["fees"]) > 0  # a point's summary carries its real fees
        assert point["excess_total_return"] == result["benchmark"]["relative"]["excess_total_return"]

    kinds = {a["kind"]: a["path"] for a in payload["artifacts"]}
    assert set(kinds) == {"sweep"}  # sweep.json and sweep.csv
    sweep_dir = lab / ".signalquarry" / "sweeps" / payload["data"]["sweep_id"]
    document = json.loads((sweep_dir / "sweep.json").read_text())
    assert document["schema"] == "signalquarry.sweep/v1" and document["sweep_id"].endswith("-sweep")
    assert document["grid"] == {"period": [150, 200]} and document["points"] == points
    assert document["pbo"] == payload["data"]["pbo"] and document["summary_only"] is False
    assert document["result_hash"].startswith("sha256:")
    rows = (sweep_dir / "sweep.csv").read_text().splitlines()
    assert rows[0] == (
        "period,configuration_hash,run_id,total_return,sharpe,max_drawdown,fills,excess_total_return"
    )
    assert [row.split(",")[0] for row in rows[1:]] == ["150", "200"]
    assert [row.split(",")[2] for row in rows[1:]] == [p["run_id"] for p in points]

    # The base configuration's point is the same run as a plain backtest.
    code, backtest = _sqy(capsys, "backtest", "--strategy", "sma-trend", "--project", str(lab))
    base = json.loads((lab / ".signalquarry" / "runs" / points[1]["run_id"] / "result.json").read_text())
    assert backtest["data"]["configuration_hash"] == points[1]["configuration_hash"]
    assert backtest["data"]["ledger_hash"] == base["ledger_hash"]
    plain = json.loads(
        (lab / next(a["path"] for a in backtest["artifacts"] if a["kind"] == "result")).read_text()
    )
    assert plain["command"] == "backtest" and plain["params"] == base["params"]
    assert plain["spec"] == base["spec"] and "label" not in plain


def test_sweep_summary_only_leaves_out_fills_and_decisions(
    lab: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _sqy(
        capsys,
        "sweep",
        "--strategy",
        "sma-trend",
        "--param",
        "period=150,200",
        "--summary-only",
        "--project",
        str(lab),
    )
    assert code == 0, payload
    for point in payload["data"]["points"]:
        run_dir = lab / ".signalquarry" / "runs" / point["run_id"]
        assert {p.name for p in run_dir.iterdir()} == {"result.json", "equity.csv", "benchmark.csv"}
    sweep_dir = lab / ".signalquarry" / "sweeps" / payload["data"]["sweep_id"]
    assert json.loads((sweep_dir / "sweep.json").read_text())["summary_only"] is True
    # A second sweep gets its own directory.
    code, again = _sqy(
        capsys, "sweep", "--strategy", "sma-trend", "--param", "period=150", "--project", str(lab)
    )
    assert again["data"]["sweep_id"] != payload["data"]["sweep_id"] and again["data"]["pbo"] is None
