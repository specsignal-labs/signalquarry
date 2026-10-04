# SPDX-License-Identifier: Apache-2.0
"""The exact report of the synthetic factor checks: names, verdicts, details and the check schedule."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

import signalquarry._internal.validation.factor_conformance as module
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.validation.conformance import CheckResult
from signalquarry._internal.validation.factor_conformance import run_factor_checks
from signalquarry.sdk import FactorCtx, Params, factor, factor_definition_of

UNIVERSE = ("SYNA", "SYNB", "SYNC", "SYNX")


class FactorParams(Params):
    period: int = 3


@factor(params=FactorParams, lookback=lambda params: params.period)
def steady(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
    del params
    close = ctx.panel("close")
    return {symbol: float(close[-1, i] / close[0, i] - 1) for i, symbol in enumerate(ctx.universe)}


def _checks(definition: Any = steady, **options: Any) -> list[CheckResult]:
    return run_factor_checks(factor_definition_of(definition), FactorParams(), **options)


def _as_tuples(results: list[CheckResult]) -> list[tuple[str, bool, str]]:
    return [(item.name, item.ok, item.detail) for item in results]


def test_a_sound_factor_reports_three_passing_checks_with_their_descriptions() -> None:
    assert _as_tuples(_checks()) == [
        ("contract", True, "finite declared-symbol scores on synthetic panels"),
        ("determinism", True, "3 repeated decision sessions"),
        ("lookahead", True, "3 cuts, 2 perturbations per cut"),
    ]


@pytest.mark.parametrize("universe", [(), ("SYNA", "SYNA"), ("SYNA", "SYNB", "SYNA")])
def test_an_empty_or_duplicated_universe_is_a_contract_failure_and_nothing_else(
    universe: tuple[str, ...],
) -> None:
    assert _as_tuples(_checks(universe=universe)) == [("contract", False, "FACTOR_UNIVERSE_INVALID")]


def test_the_default_universe_is_four_synthetic_symbols() -> None:
    seen: list[tuple[str, ...]] = []

    @factor(params=FactorParams, lookback=lambda params: params.period)
    def recording(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        del params
        seen.append(ctx.universe)
        return {symbol: 1.0 for symbol in ctx.universe}

    _checks(recording)
    assert set(seen) == {UNIVERSE}


def test_a_factor_that_fails_on_its_first_runs_reports_the_exception_as_a_contract_failure() -> None:
    @factor(params=FactorParams, lookback=lambda params: params.period)
    def broken(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        del ctx, params
        raise RuntimeError("boom")

    assert _as_tuples(_checks(broken)) == [("contract", False, "RuntimeError: boom")]

    @factor(params=FactorParams, lookback=lambda params: params.period)
    def foreign(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        del ctx, params
        return {"OUTSIDE": 1.0}

    assert _as_tuples(_checks(foreign)) == [("contract", False, "ValueError: FACTOR_SYMBOL_UNDECLARED")]


def test_a_failure_while_repeating_stops_after_the_determinism_report() -> None:
    calls = 0

    @factor(params=FactorParams, lookback=lambda params: params.period)
    def fails_on_repeat(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        nonlocal calls
        del params
        calls += 1
        if calls == 4:
            raise RuntimeError("synthetic failure")
        return {ctx.universe[0]: 1.0}

    assert _as_tuples(_checks(fails_on_repeat)) == [
        ("contract", True, "finite declared-symbol scores on synthetic panels"),
        ("determinism", False, "repeat failed: RuntimeError: synthetic failure"),
    ]


def test_a_failure_on_a_perturbed_panel_is_a_lookahead_failure() -> None:
    calls = 0

    @factor(params=FactorParams, lookback=lambda params: params.period)
    def fails_when_perturbed(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        nonlocal calls
        del params
        calls += 1
        if calls == 7:
            raise RuntimeError("synthetic failure")
        return {ctx.universe[0]: 1.0}

    assert _as_tuples(_checks(fails_when_perturbed)) == [
        ("contract", True, "finite declared-symbol scores on synthetic panels"),
        ("determinism", True, "3 repeated decision sessions"),
        ("lookahead", False, "mutated run failed: synthetic failure"),
    ]


def test_a_changed_score_names_the_first_session_and_stops() -> None:
    calls = 0
    data = synthetic_dataset(date(2018, 1, 2), date(2023, 12, 29), symbols=UNIVERSE)
    first_cut = int(len(data.sessions) * 0.45)

    @factor(params=FactorParams, lookback=lambda params: params.period)
    def drifts_after_replay(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        nonlocal calls
        del params
        calls += 1
        return {ctx.universe[0]: 1.0 if calls <= 6 else float(calls)}

    results = _checks(drifts_after_replay)
    assert _as_tuples(results) == [
        ("contract", True, "finite declared-symbol scores on synthetic panels"),
        ("determinism", True, "3 repeated decision sessions"),
        ("lookahead", False, f"score for {data.sessions[first_cut]} changed with later bars"),
    ]
    assert calls == 7  # stops at the first changed score


def test_the_check_schedule_uses_fixed_cuts_and_perturbation_ramps(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[int, float, float, int, date, date, tuple[str, ...]]] = []
    real = module.perturb_after

    def spy(dataset: Any, cut: int, first: float = 0.6, last: float = 1.7, **kwargs: Any) -> Any:
        recorded.append(
            (
                cut,
                first,
                last,
                len(dataset.sessions),
                dataset.sessions[0],
                dataset.sessions[-1],
                dataset.symbols,
            )
        )
        return real(dataset, cut, first, last, **kwargs)

    monkeypatch.setattr(module, "perturb_after", spy)
    assert all(item.ok for item in _checks())
    count = recorded[0][3]
    cuts = [int(count * fraction) for fraction in (0.45, 0.6, 0.75)]
    assert [(item[0], item[1], item[2]) for item in recorded] == [
        (cut, low, high) for cut in cuts for low, high in ((0.5, 0.8), (2.0, 1.3))
    ]
    assert {item[4] for item in recorded} == {date(2018, 1, 2)}
    assert {item[5] for item in recorded} == {date(2023, 12, 29)}
    assert {item[6] for item in recorded} == {tuple(sorted(UNIVERSE))}
    assert len(cuts) == len(set(cuts)) and 1400 < count < 1600


def test_each_decision_session_is_scored_on_a_window_that_ends_before_it() -> None:
    windows: list[tuple[date, date]] = []

    @factor(params=FactorParams, lookback=lambda params: params.period)
    def watching(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        del params
        windows.append((ctx.sessions[-1], ctx.decision_session))
        return {symbol: 1.0 for symbol in ctx.universe}

    _checks(watching)
    assert windows and all(last < decision for last, decision in windows)
    assert len({decision for _, decision in windows}) == 3
