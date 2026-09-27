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
