# SPDX-License-Identifier: Apache-2.0
"""Hand-worked position attribution, exact reconciliation and undefined exposures."""

from datetime import date
from decimal import Decimal as D

import pytest

from signalquarry._internal.validation.attribution import attribution

DAY = date(2025, 12, 31)
NEXT = date(2026, 1, 2)


def _reconciles(result):
    parts = result["reconciliation"]
    assert D(parts["total"]) == D(parts["symbols"]) + D(parts["fees"]) + D(parts["remainder"])
    for row in result["by_year"]:
        assert D(row["total"]) == D(row["symbols"]) + D(row["fees"]) + D(row["remainder"])
        assert D(row["symbols"]) == sum(map(D, row["by_symbol"].values()), D(0))
    for key in ("total", "symbols", "fees", "remainder"):
        assert D(parts[key]) == sum((D(row[key]) for row in result["by_year"]), D(0))


def test_attribution_buy_and_sell_once_with_fees_across_year_boundary():
    result = attribution(
        [DAY, NEXT],
        [D(103), D(106)],
        D(100),
        [(DAY, "AAA", D(2), D(12), D(24))],
        [(DAY, "AAA", "buy", D(2), D(10), D(1)), (NEXT, "AAA", "sell", D(2), D(14), D(1))],
    )
    assert result["by_symbol"] == [
        {
            "symbol": "AAA",
            "pnl": "8",
            "share_initial": 0.08,
            "sessions_held": 1,
            "average_weight": round(24 / 103, 6),
        }
    ]
    assert result["reconciliation"] == {"total": "6", "symbols": "8", "fees": "-2", "remainder": "0"}
    assert [row["year"] for row in result["by_year"]] == [2025, 2026]
    assert [row["symbols"] for row in result["by_year"]] == ["4", "4"]
    assert result["exposure"]["position_session_share"] == 0.5
    _reconciles(result)


def test_attribution_two_symbols_exposure_and_concentration():
    result = attribution(
        [DAY, NEXT],
        [D(100), D(102)],
        D(100),
        [
            (DAY, "AAA", D(2), D(10), D(20)),
            (DAY, "BBB", D(1), D(20), D(20)),
            (NEXT, "AAA", D(2), D(12), D(24)),
            (NEXT, "BBB", D(1), D(18), D(18)),
        ],
        [(DAY, "AAA", "buy", D(2), D(10), D(0)), (DAY, "BBB", "buy", D(1), D(20), D(0))],
    )
    assert [row["pnl"] for row in result["by_symbol"]] == ["4", "-2"]
    assert [row["sessions_held"] for row in result["by_symbol"]] == [2, 2]
    exposure = result["exposure"]
    assert exposure["average_invested_share"] == round((40 / 100 + 42 / 102) / 2, 6)
    assert exposure["average_holdings"] == exposure["maximum_holdings"] == 2
    assert exposure["average_herfindahl"] == round((0.5 + (24 / 42) ** 2 + (18 / 42) ** 2) / 2, 6)
    assert exposure["largest_weight"] == {
        "symbol": "AAA",
        "session": NEXT.isoformat(),
        "weight": round(24 / 102, 6),
    }
    _reconciles(result)


def test_attribution_split_dividend_receivable_and_rounding_stay_in_remainder():
    days = [date(2025, 1, day) for day in (6, 7, 8, 9)]
    result = attribution(
        days,
        [D(100), D(100), D("104.000001"), D("104.000001")],
        D(100),
        [
            (days[0], "AAA", D(2), D(10), D(20)),
            (days[1], "AAA", D(4), D(5), D(20)),
            (days[2], "AAA", D(4), D(5), D(20)),
        ],
        [(days[0], "AAA", "buy", D(2), D(10), D(0)), (days[3], "AAA", "sell", D(4), D(5), D(0))],
    )
    assert result["by_symbol"][0]["pnl"] == "0"
    assert result["by_symbol"][0]["sessions_held"] == 3
    assert D(result["reconciliation"]["remainder"]) == D("4.000001")
    assert "Dividends" in result["remainder_note"] and "rounding" in result["remainder_note"]
    _reconciles(result)


def test_attribution_intraday_round_trip_is_included_without_a_closing_holding():
    result = attribution(
        [DAY],
        [D(105)],
        D(100),
        [(DAY, "BBB", D(1), D(10), D(10))],
        [
            (DAY, "BBB", "buy", D(1), D(10), D(0)),
            (DAY, "AAA", "buy", D(1), D(10), D(0)),
            (DAY, "AAA", "sell", D(1), D(15), D(0)),
        ],
    )
    assert result["by_symbol"][0] == {
        "symbol": "AAA",
        "pnl": "5",
        "share_initial": 0.05,
        "sessions_held": 0,
        "average_weight": None,
    }
    _reconciles(result)


@pytest.mark.parametrize(
    ("sessions", "equity", "reason"), [([], [], "No sessions."), ([DAY], [D(100)], "No positions rows.")]
)
def test_attribution_insufficient(sessions, equity, reason):
    assert attribution(sessions, equity, D(100), [], []) == {"status": "insufficient", "reason": reason}


def test_attribution_zero_marks_nonpositive_equity_and_ties():
    result = attribution([DAY], [D(0)], D(100), [(DAY, "AAA", D(1), D(0), D(0))], [])
    assert result["by_symbol"][0]["average_weight"] is None
    assert result["exposure"]["average_invested_share"] is None
    assert result["exposure"]["average_herfindahl"] is None
    assert result["exposure"]["largest_weight"] is None
    _reconciles(result)
    tied = attribution(
        [DAY, NEXT],
        [D(100), D(100)],
        D(100),
        [(s, name, D(1), D(10), D(10)) for s in (DAY, NEXT) for name in ("BBB", "AAA")],
        [],
    )
    assert tied["exposure"]["largest_weight"]["symbol"] == "AAA"
    assert tied["exposure"]["largest_weight"]["session"] == DAY.isoformat()


@pytest.mark.parametrize(
    ("sessions", "equity", "initial"),
    [
        ([DAY], [], D(100)),
        ([NEXT, DAY], [D(1), D(1)], D(100)),
        ([DAY, DAY], [D(1), D(1)], D(100)),
        ([DAY], [D(1)], D(0)),
        ([DAY], [D(1)], D("NaN")),
        ([DAY], [D("Infinity")], D(100)),
    ],
)
def test_attribution_invalid_curve(sessions, equity, initial):
    with pytest.raises(ValueError, match="not aligned"):
        attribution(sessions, equity, initial, [], [])


@pytest.mark.parametrize(
    "row",
    [
        (NEXT, "AAA", D(1), D(10), D(10)),
        (DAY, "", D(1), D(10), D(10)),
        (DAY, "AAA", D(0), D(10), D(0)),
        (DAY, "AAA", D(1), D(-10), D(-10)),
        (DAY, "AAA", D(1), D(10), D(11)),
        (DAY, "AAA", D("NaN"), D(10), D(10)),
    ],
)
def test_attribution_invalid_positions(row):
    with pytest.raises(ValueError, match="Positions"):
        attribution([DAY], [D(100)], D(100), [row], [])


def test_attribution_duplicate_position():
    row = (DAY, "AAA", D(1), D(10), D(10))
    with pytest.raises(ValueError, match="duplicated"):
        attribution([DAY], [D(100)], D(100), [row, row], [])


@pytest.mark.parametrize(
    "row",
    [
        (NEXT, "AAA", "buy", D(1), D(10), D(0)),
        (DAY, "", "buy", D(1), D(10), D(0)),
        (DAY, "AAA", "bad", D(1), D(10), D(0)),
        (DAY, "AAA", "buy", D(0), D(10), D(0)),
        (DAY, "AAA", "buy", D(1), D(-10), D(0)),
        (DAY, "AAA", "buy", D(1), D(10), D(-1)),
        (DAY, "AAA", "buy", D(1), D("NaN"), D(0)),
    ],
)
def test_attribution_invalid_fills(row):
    with pytest.raises(ValueError, match="Fills"):
        attribution([DAY], [D(100)], D(100), [(DAY, "AAA", D(1), D(10), D(10))], [row])
