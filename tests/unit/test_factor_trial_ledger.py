# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from signalquarry._internal.contracts.factor_spec import FactorEvaluationSpecV1
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.factor_trials import (
    FactorTrialConfiguration,
    record_factor_trial,
)


def _identity(character: str) -> str:
    return f"sha256:{character * 64}"


def _configuration(
    *,
    factor: str = "a",
    dataset: str = "b",
    universe: str = "c",
    labels: str = "d",
) -> FactorTrialConfiguration:
    return FactorTrialConfiguration(
        factor_configuration_hash=_identity(factor),
        dataset_identity=_identity(dataset),
        universe_identity=_identity(universe),
        label_identity=_identity(labels),
        decision_sessions=(date(2026, 1, 2), date(2026, 1, 5)),
        evaluation=FactorEvaluationSpecV1(),
    )


def _record(
    root: Path,
    family: str,
    configuration: FactorTrialConfiguration,
) -> tuple[dict, bool]:
    return record_factor_trial(
        root,
        family=family,
        factor_id="momentum",
        configuration=configuration,
        at=datetime(2026, 1, 6, tzinfo=UTC),
        metrics={"mean_ic": 0.12, "observations": 10},
    )


def test_project_layout_factor_trials_are_idempotent_and_separate_from_strategy_trials(
    tmp_path: Path,
) -> None:
    first_config = _configuration()
    first, created = _record(tmp_path, "alpha", first_config)
    duplicate, created_duplicate = _record(tmp_path, "alpha", first_config)
    second_config = replace(first_config, dataset_identity=_identity("e"))
    second, created_second = _record(tmp_path, "alpha", second_config)

    assert created and not created_duplicate and created_second
    assert duplicate["hash"] == first["hash"]
    assert first["trial_configuration_hash"] != second["trial_configuration_hash"]
    summary = ledger.factor_trial_summary(tmp_path, "alpha")
    assert (summary["project_count"], summary["family_count"], summary["trial_count"]) == (1, 1, 2)
    assert summary["head"] == second["hash"]
    assert ledger.trial_summary(tmp_path)["project_count"] == 0
    assert (tmp_path / "evidence/factor_trials.jsonl").is_file()
    assert not (tmp_path / "evidence/trials.jsonl").exists()
    assert ledger.verify_index(tmp_path) == []


def test_per_family_factor_counts_and_index_are_project_wide(tmp_path: Path) -> None:
    (tmp_path / "signalquarry.toml").write_text('[evidence]\nlayout = "per_family"\n')
    alpha_config = _configuration()
    beta_config = _configuration(factor="f", dataset="e", universe="f", labels="0")
    alpha, _ = _record(tmp_path, "alpha", alpha_config)
    beta, _ = _record(tmp_path, "beta", beta_config)

    assert (tmp_path / "families/alpha/evidence/factor_trials.jsonl").is_file()
    assert (tmp_path / "families/beta/evidence/factor_trials.jsonl").is_file()
    assert ledger.factor_trial_summary(tmp_path, "alpha") == {
        "project_count": 2,
        "family_count": 1,
        "trial_count": 1,
        "head": ledger.project_index(tmp_path).entries()[-1]["hash"],
    }
    assert ledger.factor_trial_summary(tmp_path, "beta")["family_count"] == 1
    assert ledger.factor_trial_summary(tmp_path)["project_count"] == 2
    assert ledger.trial_summary(tmp_path)["project_count"] == 0
    assert ledger.verify_index(tmp_path) == []
    assert [entry["log"] for entry in ledger.project_index(tmp_path).entries()] == [
        "factor_trials",
        "factor_trials",
    ]
    assert alpha["family"] == "alpha" and beta["family"] == "beta"


def test_factor_log_rollback_is_detected(tmp_path: Path) -> None:
    (tmp_path / "signalquarry.toml").write_text('[evidence]\nlayout = "per_family"\n')
    _record(tmp_path, "alpha", _configuration())
    path = tmp_path / "families/alpha/evidence/factor_trials.jsonl"
    path.write_text("")
    assert ledger.verify_index(tmp_path) == ["EVIDENCE_INDEX_MISMATCH:alpha/factor_trials"]


def test_factor_project_index_tampering_is_detected(tmp_path: Path) -> None:
    (tmp_path / "signalquarry.toml").write_text('[evidence]\nlayout = "per_family"\n')
    _record(tmp_path, "alpha", _configuration())
    path = tmp_path / "evidence/project_index.jsonl"
    entry = json.loads(path.read_text())
    entry["head"] = _identity("f")
    path.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ledger.LedgerError, match="EVIDENCE_LOG_CORRUPT"):
        ledger.verify_index(tmp_path)


@pytest.mark.parametrize(
    ("family", "factor_id"),
    [("../outside", "momentum"), ("", "momentum"), ("bad/name", "momentum"), ("alpha", "../outside")],
)
def test_factor_trial_names_cannot_escape_ledger_layout(tmp_path: Path, family: str, factor_id: str) -> None:
    with pytest.raises(ValueError, match="FACTOR_TRIAL_FAMILY_INVALID"):
        record_factor_trial(
            tmp_path,
            family=family,
            factor_id=factor_id,
            configuration=_configuration(),
            at=datetime(2026, 1, 6, tzinfo=UTC),
            metrics={"mean_ic": 0.12},
        )
    assert not (tmp_path.parent / "outside").exists()


@pytest.mark.parametrize(
    ("at", "metrics", "message"),
    [
        (datetime(2026, 1, 6), {"mean_ic": 0.12}, "FACTOR_TRIAL_TIME_INVALID"),
        (datetime(2026, 1, 6, tzinfo=UTC), cast(Any, []), "FACTOR_TRIAL_METRICS_INVALID"),
    ],
)
def test_factor_trial_record_rejects_invalid_time_or_metrics(
    tmp_path: Path,
    at: datetime,
    metrics: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        record_factor_trial(
            tmp_path,
            family="alpha",
            factor_id="momentum",
            configuration=_configuration(),
            at=at,
            metrics=metrics,
        )
