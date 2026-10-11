# SPDX-License-Identifier: Apache-2.0
"""Diagnostic rendering uses the artifact's values, including missing values."""

from __future__ import annotations

from signalquarry._internal.evidence.report import diagnostic_lines


def test_all_unavailable_sections_are_omitted_and_explained() -> None:
    document = {
        name: {"status": "unavailable", "reason": f"No {name}."}
        for name in ("regimes", "costs", "folds", "parameters")
    }
    text = "\n".join(diagnostic_lines(document))
    assert (
        text
        == "Unavailable diagnostics: regimes: No regimes. costs: No costs. folds: No folds. parameters: No parameters.\n"
    )
    assert "##" not in text


def test_missing_fold_benchmark_and_parameter_scores_and_no_crossing() -> None:
    document = {
        "regimes": {"status": "unavailable", "reason": "No regimes."},
        "costs": {
            "status": "ok",
            "points": [{"multiplier": 1, "total_return": 0.1, "sharpe": None, "max_drawdown": 0.2}],
            "break_even": None,
            "note": "Descriptive only.",
        },
        "folds": {
            "status": "ok",
            "consistency": {
                "folds": 1,
                "positive": 0,
                "share_positive": 0.0,
                "ahead_of_benchmark": None,
                "best": 0.0,
                "worst": 0.0,
                "dispersion": None,
                "status": "insufficient",
            },
        },
        "parameters": {
            "status": "ok",
            "best": None,
            "metric": "sharpe",
            "neighbour_median": None,
            "plateau": None,
        },
    }
    text = "\n".join(diagnostic_lines(document))
    assert "| 1× | 10.00% | – | 20.00% |" in text
    assert "No break-even within 4× costs: the return stays positive." in text
    assert "ahead of benchmark in unavailable folds" in text and "Status: insufficient" in text
    assert "Best point: `unavailable` (– sharpe); neighbour median –; plateau –." in text
