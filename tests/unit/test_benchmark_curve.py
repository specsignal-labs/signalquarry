# SPDX-License-Identifier: Apache-2.0
"""The benchmark curve on a run's sessions, its summary and its place in the report."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.reference import buy_and_hold
from signalquarry._internal.engine.run import simulate, simulate_equity_ticks
from signalquarry._internal.evidence.report import equity_svg, render_report
from signalquarry._internal.evidence.run_spool import spool_equity_run
from signalquarry._internal.validation.benchmark import _only, benchmark_curve, benchmark_summary
from signalquarry.sdk import Ctx, Decision, Params, definition_of, strategy
from tests.helpers import Dividend, Split, dataset, spec, weekdays

SESSIONS = weekdays(date(2025, 1, 6), 4)
PRICES = {
    "AAA": {"open": [50.0, 50.0, 50.0, 50.0]},
    "BBB": {"open": [10.0, 10.0, 12.0, 12.0], "close": [10.0, 11.0, 12.0, 13.0]},
}
FREE = {"execution": {"costs": {"bps": "0"}}, "account": {"initial_cash": "1000"}}


def test_curve_follows_the_run_sessions_and_starts_uninvested() -> None:
    # The reference run has no row for the first session; there the benchmark is still cash.
    curve = benchmark_curve(spec(("AAA",), **FREE), dataset(SESSIONS, PRICES), "BBB", SESSIONS)
    assert curve.sessions == list(SESSIONS)
    assert curve.equity == [Decimal("1000"), Decimal("1100"), Decimal("1200"), Decimal("1300")]
    assert (curve.symbol, curve.fills, curve.fees) == ("BBB", 1, Decimal("0"))
    assert curve.invested_from == SESSIONS[1]
    np.testing.assert_allclose(curve.returns(Decimal("1000")), [0.0, 0.1, 1200 / 1100 - 1, 1300 / 1200 - 1])


def test_curve_on_a_later_window_matches_the_reference_run() -> None:
    data = dataset(SESSIONS, PRICES)
    base = spec(("AAA",))
    curve = benchmark_curve(base, data, "BBB", SESSIONS[2:])
    direct = buy_and_hold(base, data, "BBB", start=SESSIONS[2], end=SESSIONS[3])
    assert curve.equity == direct.equity
    assert curve.fees == sum((fill.fee for fill in direct.fills), Decimal(0)) > 0


def test_curve_without_the_symbol_is_an_engine_error() -> None:
    with pytest.raises(EngineError, match="DATASET_SYMBOLS_MISSING:ZZZ"):
        benchmark_curve(spec(("AAA",)), dataset(SESSIONS, PRICES), "ZZZ", SESSIONS)


def test_one_symbol_view_keeps_only_that_symbol_and_its_actions() -> None:
    data = dataset(
        SESSIONS,
        PRICES,
        splits=[Split("AAA", SESSIONS[2], Decimal(2)), Split("BBB", SESSIONS[3], Decimal(2))],
        dividends=[
            Dividend("AAA", SESSIONS[1], SESSIONS[1], Decimal("1")),
            Dividend("BBB", SESSIONS[2], SESSIONS[2], Decimal("1")),
        ],
    )
    view = _only(data, "BBB")
    assert view.symbols == ("BBB",) and view.sessions == data.sessions and view.source == data.source
    assert [s.symbol for s in view.splits] == ["BBB"] and [d.symbol for d in view.dividends] == ["BBB"]
    assert view.series["BBB"] is data.series["BBB"]
    assert _only(view, "BBB") is view  # already a single symbol
    assert _only(data, "ZZZ") is data  # the engine reports the missing symbol
    full = buy_and_hold(spec(("AAA",), **FREE), data, "BBB")
    assert benchmark_curve(spec(("AAA",), **FREE), data, "BBB", full.sessions).equity == full.equity


def test_summary_has_the_benchmark_metrics_and_the_relative_block() -> None:
    curve = benchmark_curve(spec(("AAA",), **FREE), dataset(SESSIONS, PRICES), "BBB", SESSIONS)
    block = benchmark_summary(curve, curve.returns(Decimal("1000")), Decimal("1000"))
    assert block["symbol"] == "BBB" and block["invested_from"] == SESSIONS[1].isoformat()
    assert block["metrics"]["total_return"] == 0.3 and block["metrics"]["fills"] == 1
    assert block["relative"] == {"observations": 4, "status": "insufficient"}


class _Flip(Params):
    pass


@strategy(params=_Flip, lookback=lambda p: 1)
def _flip(ctx: Ctx, p: _Flip) -> Decision:
    # In the market on even days, out on odd days: a buy and a sell to total.
    if ctx.decision_session.day % 2 == 1:
        return Decision.target({"BBB": 1}, "GO")
    return Decision.target({}, "WAIT")


def test_spool_totals_the_notional_bought_and_sold(tmp_path: Path) -> None:
    base, data = spec(("BBB",), **FREE), dataset(SESSIONS, PRICES)
    definition = definition_of(_flip)
    run = spool_equity_run(simulate_equity_ticks(base, definition, _Flip(), data), tmp_path)
    fills = simulate(base, definition, _Flip(), data).fills
    assert {fill.side for fill in fills} == {"buy", "sell"}
    assert run.bought == sum((f.quantity * f.price for f in fills if f.side == "buy"), Decimal(0))
    assert run.sold == sum((f.quantity * f.price for f in fills if f.side == "sell"), Decimal(0))
    assert run.bought > 0 and run.sold > 0 and run.fill_count == len(fills)


RESULT = {
    "run_id": "r1",
    "dataset_id": "synthetic",
    "evidence": {"grade": "synthetic", "claim_level": "none"},
    "metrics": {"sessions": 4, "total_return": 0.1, "max_drawdown": 0.2, "fees": "3.00"},
}
SPEC = {
    "id": "s",
    "version": "1",
    "family": "f",
    "hypothesis": {"statement": "A statement.", "falsification": "A falsification."},
}
TRIALS = {"family_count": 0, "project_count": 0}


def _render(**extra) -> str:
    return render_report(spec=SPEC, result={**RESULT, **extra}, evaluation=None, trials=TRIALS, chart=None)


def test_report_says_when_relative_statistics_or_the_position_are_unavailable() -> None:
    text = _render(
        benchmark={
            "symbol": "BBB",
            "invested_from": None,
            "metrics": {"total_return": 0.3, "max_drawdown": 0.1},
            "relative": {"observations": 4, "status": "insufficient"},
        }
    )
    assert "| Total return | 10.00% | 30.00% | -20.00% |" in text
    assert "| CAGR | – | – | – |" in text
    assert "Too few sessions (4) for statistics relative to the benchmark." in text
    assert "which could not be bought in this window." in text
    assert "### Versus" not in _render()


def test_report_tables_for_drawdowns_and_trading() -> None:
    text = _render(
        drawdowns=[
            {
                "peak": None,
                "trough": "2025-01-06",
                "recovery": None,
                "depth": 0.2,
                "sessions_to_trough": 1,
                "sessions_to_recovery": None,
            }
        ],
        activity={"status": "ok", "turnover": 1.5, "implied_holding_sessions": 168.0, "fee_drag": 0.003},
    )
    assert "| start | 2025-01-06 | not yet | 20.00% | 1 | – |" in text
    assert (
        "One-way turnover 1.50 times a year (about 168 sessions per holding); "
        "fees 3.00 in total, 0.30% of mean equity a year."
    ) in text
    silent = _render(activity={"status": "insufficient"}, drawdowns=[])
    assert "### Trading" not in silent and "### Drawdowns" not in silent
    never = _render(
        activity={"status": "ok", "turnover": 0.0, "implied_holding_sessions": None, "fee_drag": 0.0}
    )
    assert "One-way turnover 0.00 times a year; fees" in never


def test_chart_draws_the_benchmark_only_when_it_lines_up() -> None:
    points = [("2025-01-06", Decimal("100")), ("2025-01-07", Decimal("110"))]
    alone = equity_svg(points)
    assert alone.count("<polyline") == 1 and "against" not in alone
    both = equity_svg(points, benchmark=[("2025-01-06", Decimal("100")), ("2025-01-07", Decimal("150"))])
    assert both.count("<polyline") == 2 and "against benchmark" in both and ">150<" in both
    named = equity_svg(points, benchmark=points, benchmark_label="A&B")
    assert "A&amp;B" in named
    assert equity_svg(points, benchmark=points[:1]).count("<polyline") == 1
    assert equity_svg(points[:1], benchmark=points[:1]) == ""


def test_growth_chart_rebases_each_curve_and_skips_what_does_not_line_up() -> None:
    from signalquarry._internal.evidence.report import growth_svg

    days = ["2025-01-06", "2025-01-07", "2025-01-08"]
    first = list(zip(days, [Decimal("100"), Decimal("110"), Decimal("121")], strict=True))
    second = list(zip(days, [Decimal("50"), Decimal("45"), Decimal("60")], strict=True))
    svg = growth_svg([("a&b", first), ("other", second), ("short", first[:2])])
    assert svg.count("<polyline") == 2 and "a&amp;b" in svg and ">other<" in svg and "short" not in svg
    assert ">1.21<" in svg and ">0.90<" in svg  # growth of 1: 121/100 at the top, 45/50 at the bottom
    assert 'aria-label="Growth of 1 from 2025-01-06 to 2025-01-08 for 2 runs"' in svg
    assert growth_svg([("one", first[:1])]) == ""
    assert growth_svg([("zero", [(days[0], Decimal("0")), (days[1], Decimal("1"))])]) == ""
    flat = growth_svg([("flat", [(days[0], Decimal("5")), (days[1], Decimal("5"))])])
    assert flat.count("<polyline") == 1


def test_comparison_report_without_pairs_or_chart() -> None:
    from signalquarry._internal.evidence.report import render_comparison

    comparison = {
        "reference": "r1",
        "comparable": False,
        "reasons": ["COMPARE_DATASET_DIFFERS", "COMPARE_GRADE_DIFFERS"],
        "runs": [
            {
                "run_id": "r1",
                "strategy_id": "s",
                "configuration_hash": "sha256:1",
                "dataset_identity": "sha256:a",
                "grade": "historical",
                "metrics": {"total_return": 0.1},
            },
            {
                "run_id": "r2",
                "strategy_id": "s",
                "configuration_hash": "sha256:2",
                "dataset_identity": "sha256:b",
                "grade": "synthetic",
                "metrics": {},
            },
        ],
        "differences": {"r2": {"params": {}, "spec": {"execution.costs.bps": ["5", "10"]}}},
        "pairs": {},
    }
    text = render_comparison(comparison, labels={"r1": "base"}, chart=False)
    assert "grade `historical`, `synthetic`" in text and "Synthetic data" in text
    assert "**Not comparable** (COMPARE_DATASET_DIFFERS, COMPARE_GRADE_DIFFERS)" in text
    assert "| base (reference) | s | 10.00% |" in text and "| r2 | s | – |" in text
    assert "- **r2**: `execution.costs.bps`: `5` → `10`" in text
    assert "![Growth of 1]" not in text and "## Against the reference" not in text

    same = {**comparison, "differences": {"r2": {"params": {}, "spec": {}}}}
    assert "- **r2**: no recorded difference" in render_comparison(same, labels={}, chart=True)
    paired = {
        **comparison,
        "comparable": True,
        "reasons": [],
        "pairs": {
            "r2": {
                "correlation": None,
                "sharpe_difference": None,
                "sharpe_difference_90": None,
                "total_return_difference": -0.05,
                "max_drawdown_difference": 0.02,
                "sessions": 2,
            }
        },
    }
    assert "| r2 | – | – | – | -5.00% | +2.00% |" in render_comparison(paired, labels={}, chart=False)
