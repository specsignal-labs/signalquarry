# SPDX-License-Identifier: Apache-2.0
"""Exposure rendering appends descriptive values without changing older documents."""

from __future__ import annotations

from typing import Any

import pytest

from signalquarry._internal.evidence.report import diagnostic_lines


def _document() -> dict[str, Any]:
    return {
        name: {"status": "unavailable", "reason": f"No {name}."}
        for name in ("regimes", "costs", "folds", "parameters")
    }


def _exposure() -> dict[str, Any]:
    return {
        "status": "ok",
        "regression": {
            "references": [
                {"name": "SPY", "beta": 0.65, "t": 3.25, "contribution_annual": 0.05},
                {"name": "TLT", "beta": -0.1, "t": None, "contribution_annual": -0.02},
            ],
            "alpha_annual": 0.01,
            "alpha_t": 0.5,
            "r_squared": 0.6,
            "residual_volatility": 0.12,
        },
        "rolling": {
            "rows": [
                {"end": "2020-01-01", "status": "ok", "betas": {"SPY": 0.4, "TLT": -0.2}},
                {"end": "2020-02-01", "status": "collinear"},
                {"end": "2020-03-01", "status": "ok", "betas": {"SPY": 0.8, "TLT": 0.1}},
            ]
        },
        "note": "References chosen after the run; in-sample associations with passive positions.",
    }


def test_without_exposure_is_byte_identical() -> None:
    assert "\n".join(diagnostic_lines(_document())) == (
        "Unavailable diagnostics: regimes: No regimes. costs: No costs. "
        "folds: No folds. parameters: No parameters.\n"
    )


def test_exposure_table_summary_ranges_and_note() -> None:
    document = _document()
    document["exposure"] = _exposure()
    text = "\n".join(diagnostic_lines(document))
    assert "## Exposure to reference series\n\n| Reference | Beta | t | Annual contribution |" in text
    assert "| SPY | 0.650000 | 3.25 | 5.00% |" in text
    assert "| TLT | -0.100000 | – | -2.00% |" in text
    assert "Annual alpha 1.00% (t 0.50); R² 0.600000; residual volatility 12.00%." in text
    assert "Rolling beta SPY: 0.400000 to 0.800000." in text
    assert "Rolling beta TLT: -0.200000 to 0.100000." in text
    assert document["exposure"]["note"] in text
    assert text.index("## Exposure") < text.index("Unavailable diagnostics:")
    assert "skill" not in text.lower()


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("unavailable", "No passive data for SPY."),
        ("insufficient", "Too few sessions."),
        ("collinear", "The references move together too closely to separate."),
    ],
)
def test_non_ok_status_on_existing_unavailable_line(status: str, reason: str) -> None:
    document = _document()
    document["exposure"] = {"status": status}
    if status == "unavailable":
        document["exposure"]["reason"] = reason
    text = "\n".join(diagnostic_lines(document))
    assert text.count("Unavailable diagnostics:") == 1
    assert f"exposure: {reason}" in text
    assert "## Exposure" not in text


def test_missing_values_render_as_en_dash() -> None:
    document = _document()
    document["exposure"] = {
        "status": "ok",
        "regression": {"references": [{"name": "SPY"}]},
    }
    text = "\n".join(diagnostic_lines(document))
    assert "| SPY | – | – | – |" in text
    assert "Annual alpha – (t –); R² –; residual volatility –." in text
    assert "Rolling beta SPY: – to –." in text
