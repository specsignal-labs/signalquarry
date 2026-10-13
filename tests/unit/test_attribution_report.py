# SPDX-License-Identifier: Apache-2.0
"""Attribution report ordering, omitted sections and undefined exposure rendering."""

from datetime import date
from decimal import Decimal as D

from signalquarry._internal.evidence.report import diagnostic_lines
from signalquarry._internal.validation.attribution import attribution


def _base():
    return {
        name: {"status": "unavailable", "reason": f"No {name}."}
        for name in ("regimes", "costs", "folds", "parameters")
    }


def test_attribution_report_top_ten_rest_and_exact_currency_totals():
    day = date(2025, 1, 6)
    block = attribution(
        [day], [D(166)], D(100), [(day, f"S{i:02}", D(1), D(i), D(i)) for i in range(1, 12)], []
    )
    doc = _base()
    original = "\n".join(diagnostic_lines(doc))
    doc["attribution"] = block
    text = "\n".join(diagnostic_lines(doc))
    assert "## Attribution" in text and "Descriptive and in-sample" in text
    assert text.index("| S11 |") < text.index("| S10 |") < text.index("| S02 |")
    assert "| S01 |" not in text and "Remaining 1 symbols: P&L 1.00." in text
    assert "Total P&L 66 = symbols 66 + fees 0 + remainder 0" in text
    assert "| 2025 | 66 | 0 | 0 | 66 |" in text
    assert "Herfindahl" in text and "largest single weight" in text
    doc.pop("attribution")
    assert "\n".join(diagnostic_lines(doc)) == original


def test_attribution_report_unavailable_and_nonpositive_equity():
    doc = _base()
    doc["attribution"] = {"status": "insufficient", "reason": "No positions rows."}
    text = "\n".join(diagnostic_lines(doc))
    assert "## Attribution" not in text and "attribution: No positions rows." in text
    day = date(2025, 1, 6)
    doc["attribution"] = attribution([day], [D(0)], D(100), [(day, "AAA", D(1), D(0), D(0))], [])
    text = "\n".join(diagnostic_lines(doc))
    assert "largest single weight unavailable." in text and "Average invested share –" in text
