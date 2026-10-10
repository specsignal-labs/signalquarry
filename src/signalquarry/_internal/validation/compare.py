# SPDX-License-Identifier: Apache-2.0
"""Pure comparisons of aligned backtest returns and parsed result documents."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from signalquarry._internal.validation.metrics import SESSIONS_PER_YEAR
from signalquarry._internal.validation.stats import moments, paired_block_bootstrap_sharpe_difference


@dataclass(frozen=True)
class RunSeries:
    document: Mapping[str, Any]
    sessions: tuple[date, ...]
    returns: np.ndarray


def configuration_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Sorted dotted-path differences; missing keys use None and lists stay whole."""
    differences: dict[str, list[Any]] = {}

    def visit(a: Mapping[str, Any], b: Mapping[str, Any], prefix: str) -> None:
        for key in a.keys() | b.keys():
            path = f"{prefix}.{key}" if prefix else key
            left_value, right_value = a.get(key), b.get(key)
            if key not in a or key not in b:
                differences[path] = [left_value, right_value]
            elif isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
                visit(left_value, right_value, path)
            elif left_value != right_value:
                differences[path] = [left_value, right_value]

    visit(left, right, "")
    return dict(sorted(differences.items()))


def comparability(runs: Sequence[RunSeries]) -> dict[str, Any]:
    """Report sorted reasons that prevent comparisons on the supplied sessions."""
    reasons = []
    if runs:
        reference = runs[0]
        if any(
            run.document["dataset_identity"] != reference.document["dataset_identity"] for run in runs[1:]
        ):
            reasons.append("COMPARE_DATASET_DIFFERS")
        if any(run.sessions != reference.sessions for run in runs[1:]):
            reasons.append("COMPARE_WINDOW_DIFFERS")
        accounts = [run.document["spec"].get("account") for run in runs if "spec" in run.document]
        if accounts and any(account != accounts[0] for account in accounts[1:]):
            reasons.append("COMPARE_ACCOUNT_DIFFERS")
        if any(
            run.document["evidence"]["grade"] != reference.document["evidence"]["grade"] for run in runs[1:]
        ):
            reasons.append("COMPARE_GRADE_DIFFERS")
    return {"comparable": not reasons, "reasons": sorted(reasons)}


def compare_runs(
    runs: Sequence[RunSeries], *, block: int = 20, samples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Compare each run against the first, retaining pairing and configuration differences.

    Pair statistics use only jointly finite returns; ``sessions`` counts those rows.
    Result-document metrics are retained as supplied.
    """
    if len(runs) < 2:
        raise ValueError("COMPARE_NEEDS_TWO_RUNS")
    seen = set()
    for run in runs:
        run_id = run.document["run_id"]
        if run_id in seen:
            raise ValueError(f"COMPARE_RUN_DUPLICATE:{run_id}")
        seen.add(run_id)

    reference = runs[0]
    verdict = comparability(runs)
    differences = {}
    pairs = {}
    for run in runs[1:]:
        run_id = run.document["run_id"]
        differences[run_id] = {
            part: configuration_diff(reference.document[part], run.document[part])
            if part in reference.document and part in run.document
            else {}
            for part in ("params", "spec")
        }
        if not verdict["comparable"]:
            continue

        low, high = paired_block_bootstrap_sharpe_difference(
            run.returns, reference.returns, block=block, samples=samples, seed=seed
        )
        values = np.asarray(run.returns, dtype=np.float64)
        other = np.asarray(reference.returns, dtype=np.float64)
        finite = np.isfinite(values) & np.isfinite(other)
        values, other = values[finite], other[finite]
        correlation = None
        if len(values) > 1 and values.std(ddof=1) > 0 and other.std(ddof=1) > 0:
            correlation = round(float(np.corrcoef(values, other)[0, 1]), 6)
        sharpe, reference_sharpe = moments(values).sharpe, moments(other).sharpe
        sharpe_difference = None
        interval = None
        if math.isfinite(sharpe) and math.isfinite(reference_sharpe):
            annualization = math.sqrt(SESSIONS_PER_YEAR)
            sharpe_difference = round((sharpe - reference_sharpe) * annualization, 6)
            if math.isfinite(low) and math.isfinite(high):
                interval = [round(low * annualization, 6), round(high * annualization, 6)]
        pairs[run_id] = {
            "correlation": correlation,
            "sharpe_difference": sharpe_difference,
            "sharpe_difference_90": interval,
            "total_return_difference": round(
                run.document["metrics"]["total_return"] - reference.document["metrics"]["total_return"], 8
            ),
            "max_drawdown_difference": round(
                run.document["metrics"]["max_drawdown"] - reference.document["metrics"]["max_drawdown"], 8
            ),
            "sessions": len(values),
        }

    return {
        "reference": reference.document["run_id"],
        **verdict,
        "runs": [
            {
                "run_id": run.document["run_id"],
                "strategy_id": run.document["strategy_id"],
                "configuration_hash": run.document["configuration_hash"],
                "dataset_identity": run.document["dataset_identity"],
                "grade": run.document["evidence"]["grade"],
                "metrics": run.document["metrics"],
            }
            for run in runs
        ],
        "differences": differences,
        "pairs": pairs,
    }
