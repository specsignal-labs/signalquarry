# SPDX-License-Identifier: Apache-2.0
"""Reference cases for walk-forward windows, reported statistics and gates."""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.backtest import BacktestResult
from signalquarry._internal.validation import evaluate as evaluation
from signalquarry._internal.validation.stats import ReturnMoments
from signalquarry.sdk import definition_of
from tests.helpers import spec, weekdays
from tests.unit.test_engine import SmaP, sma_trend


def _result(sessions: list[date]) -> BacktestResult:
    return BacktestResult(
        sessions=sessions,
        equity=[Decimal("10000")] * len(sessions),
        cash=[Decimal("10000")] * len(sessions),
        decisions=[],
        fills=[],
        positions={},
        warnings=[],
        dataset_identity="synthetic-reference",
    )


def test_fold_window_includes_both_boundary_sessions() -> None:
    sessions = [date(2025, 1, day) for day in (2, 3, 6, 7)]
    selected = evaluation._window(
        _result(sessions), np.array([0.01, 0.02, 0.03, 0.04]), sessions[1], sessions[2]
    )
    assert selected.shape == (2,)
    np.testing.assert_array_equal(selected, [0.02, 0.03])


def test_annual_interval_uses_252_sessions_and_rejects_nonfinite_bounds() -> None:
    assert evaluation._annual_interval((0.12345, 0.23456)) == [1.9597, 3.7235]
    assert evaluation._annual_interval((float("nan"), 0.1)) is None
    assert evaluation._annual_interval((0.1, float("inf"))) is None


def test_fold_statistics_match_a_compounded_return_path() -> None:
    returns = np.array([0.1, -0.1, 0.1, -0.2])
    actual = evaluation._stats(returns)
    assert set(actual) == {"sessions", "total_return", "sharpe_daily", "sharpe_annual", "max_drawdown"}
    assert actual["sessions"] == 4
    assert actual["total_return"] == pytest.approx(-0.1288, abs=1e-12)
    assert actual["sharpe_daily"] == -0.16666667
    assert actual["sharpe_annual"] == -2.645751
    assert actual["max_drawdown"] == 0.208
    assert evaluation._stats(np.array([])) == {
        "sessions": 0,
        "total_return": 0.0,
        "sharpe_daily": None,
        "sharpe_annual": None,
        "max_drawdown": 0.0,
    }


@pytest.mark.parametrize("failed_gate", ["G1_sample", "G2_walk_forward", "G3_stress"])
def test_claim_level_requires_each_prerequisite_gate(failed_gate: str) -> None:
    gates = {
        name: {"ok": name != failed_gate}
        for name in ("G1_sample", "G2_walk_forward", "G3_stress", "G4_holdout")
    }
    assert (
        evaluation.claim_level(
            evaluation.Evaluation(gates=gates), grade="historical", frozen=True, conformance_ok=True
        )
        == "in_sample"
    )


def test_options_grade_caps_claim_below_holdout_even_if_all_gates_pass() -> None:
    gates = {name: {"ok": True} for name in ("G1_sample", "G2_walk_forward", "G3_stress", "G4_holdout")}
    assert (
        evaluation.claim_level(
            evaluation.Evaluation(gates=gates),
            grade="low_evidence_options",
            frozen=True,
            conformance_ok=True,
        )
        == "walk_forward"
    )


def test_evaluation_clips_holdout_and_keeps_sample_and_walk_forward_gates_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = weekdays(date(2015, 1, 5), 2600)
    pre_end = date(2023, 12, 31)
    seen_calls: list[dict[str, object]] = []
    seen_trial_counts: list[int] = []

    def simulate_stub(*_args: object, end: date | None = None, **kwargs: object) -> BacktestResult:
        seen_calls.append({"end": end, **kwargs})
        return _result([day for day in sessions if end is None or day <= end])

    def dsr_stub(_moments: object, trials: int, _variance: float) -> float:
        seen_trial_counts.append(trials)
        return 0.2

    monkeypatch.setattr(evaluation, "simulate", simulate_stub)
    monkeypatch.setattr(evaluation, "probabilistic_sharpe", lambda _moments: 0.99)
    monkeypatch.setattr(evaluation, "deflated_sharpe", dsr_stub)
    monkeypatch.setattr(evaluation, "min_track_record_length", lambda _moments: 10.5)
    monkeypatch.setattr(evaluation, "block_bootstrap_sharpe", lambda _returns: (0.12345, 0.23456))

    outcome = evaluation.evaluate(
        spec(("SPY",), evaluation={"walk_forward": {"train_months": 36, "test_months": 6}}),
        definition_of(sma_trend),
        SmaP(),
        Dataset(sessions, {}),
        holdout_start=date(2024, 1, 1),
        project_trials=0,
        sharpe_variance=0.01,
    )

    assert len(seen_calls) == 3  # base, doubled costs, and one-session delay
    assert all(call["end"] == pre_end for call in seen_calls)
    assert any(call.get("cost_multiplier") == Decimal(2) for call in seen_calls)
    assert any(call.get("delay_sessions") == 1 for call in seen_calls)
    assert outcome.base is not None and outcome.base.sessions[-1] <= pre_end
    assert outcome.folds[0]["start"] == "2018-01-05"
    assert outcome.folds[0]["end"] == "2018-07-04"
    assert all(
        date.fromisoformat(right["start"]).toordinal() == date.fromisoformat(left["end"]).toordinal() + 1
        for left, right in zip(outcome.folds, outcome.folds[1:], strict=False)
    )
    assert len(outcome.folds) >= 6
    assert outcome.oos_returns and all(value == 0.0 for value in outcome.oos_returns)
    assert set(outcome.trial_moments) == {"n", "sharpe", "skew", "kurtosis"}
    assert seen_trial_counts == [1]
    assert outcome.oos["project_trials"] == 0
    assert outcome.oos["psr"] == 0.99 and outcome.oos["dsr"] == 0.2
    assert outcome.oos["min_track_record_sessions"] == 10.5
    assert outcome.oos["sharpe_annual_90"] == [1.9597, 3.7235]
    assert outcome.gates["G1_sample"] == {
        "ok": False,
        "years": 8.99,
        "rebalancing_sessions": 0,
    }
    assert outcome.gates["G2_walk_forward"]["ok"] is False
    assert outcome.gates["G3_stress"]["ok"] is False
    assert set(outcome.gates["G3_stress"]["scenarios"]) == {"costs_x2", "delay_1"}
    assert math.isfinite(outcome.oos["sharpe_annual_90"][0])


@pytest.mark.parametrize(
    ("base_sharpe", "stress_sharpe", "stress_return", "expected"),
    [
        (1.0, 0.5, 0.1, True),  # exactly half the base Sharpe passes
        (1.0, 0.5, 0.0, False),  # zero net return fails
        (0.0, 0.5, 0.1, False),  # no positive base Sharpe fails
        (float("nan"), 0.5, 0.1, False),  # nonfinite base Sharpe falls back to zero
        (1.0, None, 0.1, False),  # missing stressed Sharpe fails
    ],
)
def test_stress_gate_requires_each_condition(
    monkeypatch: pytest.MonkeyPatch,
    base_sharpe: float,
    stress_sharpe: float | None,
    stress_return: float,
    expected: bool,
) -> None:
    sessions = weekdays(date(2015, 1, 5), 2600)
    current: dict[str, bool] = {"stress": False}
    seen_calls: list[dict[str, object]] = []
    recorded_chains: dict[tuple[str, date], dict[str, tuple[Decimal, Decimal]]] = {}

    def simulate_stub(*_args: object, end: date | None = None, **kwargs: object) -> BacktestResult:
        current["stress"] = "cost_multiplier" in kwargs or "delay_sessions" in kwargs
        seen_calls.append(kwargs)
        return _result([day for day in sessions if end is None or day <= end])

    def stats_stub(returns: np.ndarray) -> dict[str, object]:
        return {
            "sessions": len(returns),
            "total_return": stress_return if current["stress"] else 0.1,
            "sharpe_daily": stress_sharpe if current["stress"] else base_sharpe,
            "sharpe_annual": None,
            "max_drawdown": 0.25,
        }

    monkeypatch.setattr(evaluation, "simulate", simulate_stub)
    monkeypatch.setattr(evaluation, "_stats", stats_stub)
    monkeypatch.setattr(
        evaluation, "moments", lambda returns: ReturnMoments(len(returns), base_sharpe, 0.0, 3.0)
    )
    monkeypatch.setattr(evaluation, "block_bootstrap_sharpe", lambda _returns: (float("nan"),) * 2)
    outcome = evaluation.evaluate(
        spec(("SPY",)),
        definition_of(sma_trend),
        SmaP(),
        Dataset(sessions, {}),
        holdout_start=date(2024, 1, 1),
        project_trials=1,
        sharpe_variance=0.01,
        recorded_chains=recorded_chains,
    )
    assert outcome.gates["G3_stress"]["ok"] is expected
    assert all(item["ok"] is expected for item in outcome.gates["G3_stress"]["scenarios"].values())
    assert len(seen_calls) == 3
    assert all(call["recorded_chains"] is recorded_chains for call in seen_calls)


def test_evaluation_accepts_exact_sample_and_walk_forward_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = tuple(day for day in weekdays(date(2018, 1, 1), 1600) if day <= date(2023, 12, 31))
    seen_ends: list[date | None] = []

    def simulate_stub(*_args: object, end: date | None = None, **_kwargs: object) -> BacktestResult:
        seen_ends.append(end)
        result = _result([day for day in sessions if end is None or day <= end])
        result.fills = [SimpleNamespace(session=day) for day in sessions[:30]]
        return result

    monkeypatch.setattr(evaluation, "simulate", simulate_stub)
    monkeypatch.setattr(evaluation, "probabilistic_sharpe", lambda _moments: 0.95)
    monkeypatch.setattr(evaluation, "deflated_sharpe", lambda *_args: 0.95)
    monkeypatch.setattr(evaluation, "block_bootstrap_sharpe", lambda _returns: (float("nan"),) * 2)
    outcome = evaluation.evaluate(
        spec(("SPY",), evaluation={"walk_forward": {"train_months": 36, "test_months": 6}}),
        definition_of(sma_trend),
        SmaP(),
        Dataset(sessions, {}),
        holdout_start=date(2024, 1, 1),
        project_trials=1,
        sharpe_variance=0.01,
        stress=False,
    )
    assert seen_ends == [date(2023, 12, 31)]
    assert outcome.gates["G1_sample"]["ok"] is True
    assert outcome.gates["G1_sample"]["rebalancing_sessions"] == 30
    assert outcome.gates["G2_walk_forward"] == {
        "ok": True,
        "folds": 6,
        "psr": 0.95,
        "dsr": 0.95,
    }
    assert outcome.folds[-1]["end"] == "2023-12-31"


def test_no_holdout_uses_the_last_available_session(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = tuple(weekdays(date(2020, 1, 2), 100))
    seen_ends: list[date | None] = []

    def simulate_stub(*_args: object, end: date | None = None, **_kwargs: object) -> BacktestResult:
        seen_ends.append(end)
        return _result([day for day in sessions if end is None or day <= end])

    monkeypatch.setattr(evaluation, "simulate", simulate_stub)
    monkeypatch.setattr(evaluation, "block_bootstrap_sharpe", lambda _returns: (float("nan"),) * 2)
    evaluation.evaluate(
        spec(("SPY",)),
        definition_of(sma_trend),
        SmaP(),
        Dataset(sessions, {}),
        holdout_start=None,
        project_trials=1,
        sharpe_variance=0.01,
        stress=False,
    )
    assert seen_ends == [sessions[-1]]


def test_holdout_without_complete_training_folds_uses_zero_reference_drawdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = tuple(weekdays(date(2025, 1, 2), 60))

    def simulate_stub(*_args: object, end: date | None = None, **_kwargs: object) -> BacktestResult:
        return _result([day for day in sessions if end is None or day <= end])

    monkeypatch.setattr(evaluation, "simulate", simulate_stub)
    monkeypatch.setattr(evaluation, "probabilistic_sharpe", lambda _moments: float("nan"))
    monkeypatch.setattr(evaluation, "deflated_sharpe", lambda *_args: float("nan"))
    monkeypatch.setattr(evaluation, "block_bootstrap_sharpe", lambda _returns: (float("nan"),) * 2)
    monkeypatch.setattr(
        evaluation,
        "_stats",
        lambda returns: {
            "sessions": len(returns),
            "total_return": 0.0,
            "sharpe_daily": 0.1,
            "sharpe_annual": 1.0,
            "max_drawdown": 0.0,
        },
    )
    outcome = evaluation.evaluate(
        spec(("SPY",)),
        definition_of(sma_trend),
        SmaP(),
        Dataset(sessions, {}),
        holdout_start=date(2025, 2, 1),
        project_trials=1,
        sharpe_variance=0.01,
        stress=False,
        open_holdout=True,
    )
    assert outcome.folds == []
    assert outcome.oos["psr"] is None and outcome.oos["dsr"] is None
    assert outcome.gates["G4_holdout"]["worst_walk_forward_drawdown"] == 0.0


@pytest.mark.parametrize(
    ("holdout_sharpe", "holdout_drawdown", "expected"),
    [
        (0.5, 0.375, True),  # exact 1.5x worst walk-forward drawdown passes
        (0.5, 0.4, False),
        (0.0, 0.1, False),
        (None, 0.1, False),
    ],
)
def test_holdout_gate_uses_full_window_and_strict_sharpe(
    monkeypatch: pytest.MonkeyPatch,
    holdout_sharpe: float | None,
    holdout_drawdown: float,
    expected: bool,
) -> None:
    sessions = weekdays(date(2015, 1, 5), 2600)
    holdout_start = date(2024, 1, 1)
    current: dict[str, bool] = {"holdout": False}
    seen_calls: list[dict[str, object]] = []
    recorded_chains: dict[tuple[str, date], dict[str, tuple[Decimal, Decimal]]] = {}

    def simulate_stub(*_args: object, end: date | None = None, **kwargs: object) -> BacktestResult:
        current["holdout"] = end is None
        seen_calls.append({"end": end, **kwargs})
        return _result([day for day in sessions if end is None or day <= end])

    def stats_stub(returns: np.ndarray) -> dict[str, object]:
        return {
            "sessions": len(returns),
            "total_return": 0.1,
            "sharpe_daily": holdout_sharpe if current["holdout"] else 1.0,
            "sharpe_annual": None,
            "max_drawdown": holdout_drawdown if current["holdout"] else 0.25,
        }

    monkeypatch.setattr(evaluation, "simulate", simulate_stub)
    monkeypatch.setattr(evaluation, "_stats", stats_stub)
    monkeypatch.setattr(evaluation, "block_bootstrap_sharpe", lambda _returns: (float("nan"),) * 2)
    outcome = evaluation.evaluate(
        spec(("SPY",)),
        definition_of(sma_trend),
        SmaP(),
        Dataset(sessions, {}),
        holdout_start=holdout_start,
        project_trials=1,
        sharpe_variance=0.01,
        stress=False,
        open_holdout=True,
        recorded_chains=recorded_chains,
    )
    holdout = outcome.gates["G4_holdout"]
    assert holdout["ok"] is expected
    assert holdout["worst_walk_forward_drawdown"] == 0.25
    assert holdout["sessions"] == sum(day >= holdout_start for day in sessions)
    assert [call["end"] for call in seen_calls] == [date(2023, 12, 31), None]
    assert all(call["recorded_chains"] is recorded_chains for call in seen_calls)
