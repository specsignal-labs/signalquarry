# SPDX-License-Identifier: Apache-2.0
"""Reference cases for paired uncertainty and pure run comparisons."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from signalquarry._internal.validation.compare import (
    RunSeries,
    comparability,
    compare_runs,
    configuration_diff,
)
from signalquarry._internal.validation.stats import paired_block_bootstrap_sharpe_difference


def make_run(run_id: str, returns: np.ndarray | None = None, **fields: Any) -> RunSeries:
    values = np.array([0.01, 0.02, -0.01, 0.03]) if returns is None else returns
    document = {
        "run_id": run_id,
        "strategy_id": "example",
        "configuration_hash": f"configuration-{run_id}",
        "dataset_identity": {"manifest_hash": "synthetic-example"},
        "evidence": {"grade": "verified", "claim_level": "in_sample"},
        "metrics": {"total_return": 0.05, "max_drawdown": 0.01, "sessions": len(values)},
        "params": {"lookback": 5},
        "spec": {"account": {"initial_cash": "100"}, "execution": {"costs": {"bps": "5"}}},
        **fields,
    }
    sessions = tuple(date(2025, 1, 2) + timedelta(days=k) for k in range(len(values)))
    return RunSeries(document, sessions, values)


def test_paired_identical_series_have_exactly_zero_interval() -> None:
    values = np.array([0.01, 0.02, -0.01, 0.03])
    assert paired_block_bootstrap_sharpe_difference(values, values, block=2) == (0.0, 0.0)


def test_paired_positive_shift_matches_hand_worked_percentiles() -> None:
    # PCG64(7)'s eight draws have stdevs sqrt(17)/200 (three draws),
    # sqrt(3)/75 (one), sqrt(105)/600 (three), and sqrt(3)/300 (one).
    # A +0.01 shift changes each Sharpe by 0.01 / that unchanged stdev.
    values = np.array([0.01, 0.02, -0.01, 0.03])
    expected = (
        0.65 * math.sqrt(3) / 4 + 0.35 * 2 / math.sqrt(17),
        0.35 * 6 / math.sqrt(105) + 0.65 * math.sqrt(3),
    )
    interval = paired_block_bootstrap_sharpe_difference(values + 0.01, values, block=2, samples=8, seed=7)
    assert interval == pytest.approx(expected, abs=1e-12)
    assert interval[0] > 0


def test_swapping_paired_arguments_negates_and_reverses_interval() -> None:
    values = np.array([0.01, 0.02, -0.01, 0.03])
    other = np.array([-0.01, 0.03, 0.02, 0.04])
    low, high = paired_block_bootstrap_sharpe_difference(values, other, block=2, seed=9)
    swapped = paired_block_bootstrap_sharpe_difference(other, values, block=2, seed=9)
    assert swapped == pytest.approx((-high, -low), abs=1e-12)


def test_pairing_is_much_narrower_than_independent_resampling() -> None:
    # Nearly identical paths share a large slowly varying component; common draws
    # cancel it, whereas independent draws retain its substantial uncertainty.
    values = np.sin(np.arange(200) * 0.1) * 0.01 + 0.001
    other = values + 0.00001
    paired_low, paired_high = paired_block_bootstrap_sharpe_difference(
        other, values, block=10, samples=400, seed=4
    )
    rng = np.random.Generator(np.random.PCG64(4))
    independent = []
    for _ in range(400):
        left_starts, right_starts = rng.integers(0, 191, size=(2, 20))
        left = np.concatenate([other[start : start + 10] for start in left_starts])
        right = np.concatenate([values[start : start + 10] for start in right_starts])
        independent.append(left.mean() / left.std(ddof=1) - right.mean() / right.std(ddof=1))
    independent_low, independent_high = np.percentile(independent, [5, 95])
    assert independent_high - independent_low > 100 * (paired_high - paired_low)


def test_paired_bootstrap_rejects_unequal_lengths_before_filtering() -> None:
    with pytest.raises(ValueError, match="^PAIRED_RETURNS_NOT_ALIGNED$"):
        paired_block_bootstrap_sharpe_difference(np.array([0.01, np.nan]), np.array([0.01]))


def test_paired_bootstrap_requires_exactly_two_blocks() -> None:
    short = np.array([0.01, 0.02, 0.03])
    assert all(math.isnan(value) for value in paired_block_bootstrap_sharpe_difference(short, short, block=2))
    enough = np.array([0.01, 0.02, 0.03, 0.04])
    assert paired_block_bootstrap_sharpe_difference(enough, enough, block=2) == (0.0, 0.0)


def test_paired_bootstrap_drops_nonfinite_rows_jointly() -> None:
    values = np.array([0.01, np.nan, 0.02, 0.8, -0.01, np.inf, 0.03, 0.9])
    other = np.array([0.02, 0.7, 0.03, np.inf, 0.0, 0.6, 0.04, -np.inf])
    clean = np.array([0.01, 0.02, -0.01, 0.03])
    assert paired_block_bootstrap_sharpe_difference(values, other, block=2, seed=7) == (
        paired_block_bootstrap_sharpe_difference(clean, clean + 0.01, block=2, seed=7)
    )


def test_paired_bootstrap_minimum_length_is_after_joint_filtering() -> None:
    values = np.array([0.01, 0.02, np.inf, 0.03])
    other = np.array([0.02, 0.03, 0.04, 0.05])
    assert all(
        math.isnan(value) for value in paired_block_bootstrap_sharpe_difference(values, other, block=2)
    )


@pytest.mark.parametrize("flat_side", ["left", "right", "both"])
def test_paired_bootstrap_flat_samples_count_as_zero_sharpe(flat_side: str) -> None:
    flat = np.zeros(4)
    variable = np.array([1.0, 2.0, 1.0, 2.0])
    left = variable if flat_side == "right" else flat
    right = variable if flat_side == "left" else flat
    # Every length-2 block has [1,2] or [2,1]; stdev=sqrt(1/3).
    sign = {"left": -1, "right": 1, "both": 0}[flat_side]
    expected = sign * 1.5 * math.sqrt(3)
    assert paired_block_bootstrap_sharpe_difference(left, right, block=2) == pytest.approx(
        (expected, expected), abs=1e-12
    )


def test_paired_bootstrap_truncates_the_last_block_to_original_length() -> None:
    # PCG64(7) selects starts (3,2,2); truncated sample is [4,5,3,4,3].
    # Its sample variance is 0.7, so a +1 shift changes Sharpe by 1/sqrt(0.7).
    values = np.arange(1, 6)
    expected = 1 / math.sqrt(0.7)
    assert paired_block_bootstrap_sharpe_difference(values + 1, values, block=2, samples=1, seed=7) == (
        pytest.approx((expected, expected), abs=1e-12)
    )


def test_paired_bootstrap_defaults_are_deterministic() -> None:
    values = np.sin(np.arange(80) * 0.1) * 0.01
    assert paired_block_bootstrap_sharpe_difference(values + 0.001, values) == (
        paired_block_bootstrap_sharpe_difference(values + 0.001, values, block=20, samples=1000, seed=0)
    )


def test_configuration_diff_flattens_nested_paths_and_sorts_keys() -> None:
    left = {"execution": {"costs": {"bps": "5", "fixed": "0"}}, "z": 1, "a": {"b": 1}, "a-b": 1}
    right = {"a-b": 2, "a": {"b": 2}, "z": 2, "execution": {"costs": {"bps": "10", "fixed": "0"}}}
    result = configuration_diff(left, right)
    assert result == {"a-b": [1, 2], "a.b": [1, 2], "execution.costs.bps": ["5", "10"], "z": [1, 2]}
    assert list(result) == sorted(result)


def test_configuration_diff_reports_missing_keys_on_either_side() -> None:
    assert configuration_diff({"params": {"left": 1}}, {"params": {"right": 2}}) == {
        "params.left": [1, None],
        "params.right": [None, 2],
    }


def test_configuration_diff_preserves_lists_whole() -> None:
    assert configuration_diff({"data": {"symbols": ["A", "B"]}}, {"data": {"symbols": ["B", "A"]}}) == {
        "data.symbols": [["A", "B"], ["B", "A"]]
    }


def test_configuration_diff_reports_missing_objects_and_explicit_null() -> None:
    assert configuration_diff({"account": {}, "null": None}, {"other": None}) == {
        "account": [{}, None],
        "null": [None, None],
        "other": [None, None],
    }


def test_configuration_diff_reports_object_to_scalar_changes() -> None:
    assert configuration_diff({"a": {"b": 1}}, {"a": 2}) == {"a": [{"b": 1}, 2]}


def test_configuration_diff_equal_objects_are_empty() -> None:
    value = {"a": {"b": 1}, "c": [1, {"d": 2}], "e": None}
    assert configuration_diff(value, value) == {}
    assert configuration_diff({}, {}) == {}


def test_comparability_all_equal_accepts_multiple_runs() -> None:
    assert comparability([make_run("a"), make_run("b"), make_run("c")]) == {
        "comparable": True,
        "reasons": [],
    }


def test_comparability_dataset_reason_in_isolation() -> None:
    assert comparability([make_run("a"), make_run("b", dataset_identity="different")]) == {
        "comparable": False,
        "reasons": ["COMPARE_DATASET_DIFFERS"],
    }


def test_comparability_checks_exact_session_tuple_not_just_endpoints() -> None:
    other = make_run("b")
    sessions = (other.sessions[0], other.sessions[2], other.sessions[1], other.sessions[3])
    assert comparability([make_run("a"), replace(other, sessions=sessions)]) == {
        "comparable": False,
        "reasons": ["COMPARE_WINDOW_DIFFERS"],
    }


def test_comparability_account_reason_in_isolation() -> None:
    assert comparability([make_run("a"), make_run("b", spec={"account": {"initial_cash": "200"}})]) == {
        "comparable": False,
        "reasons": ["COMPARE_ACCOUNT_DIFFERS"],
    }


def test_comparability_grade_reason_in_isolation() -> None:
    assert comparability([make_run("a"), make_run("b", evidence={"grade": "unverified"})]) == {
        "comparable": False,
        "reasons": ["COMPARE_GRADE_DIFFERS"],
    }


def test_comparability_does_not_gate_on_claim_level_or_execution_costs() -> None:
    other = make_run(
        "b",
        evidence={"grade": "verified", "claim_level": "walk_forward"},
        spec={"account": {"initial_cash": "100"}, "execution": {"costs": {"bps": "10"}}},
    )
    assert comparability([make_run("a"), other]) == {"comparable": True, "reasons": []}


def test_comparability_accounts_are_checked_even_if_reference_has_no_spec() -> None:
    reference = make_run("a")
    document = {key: value for key, value in reference.document.items() if key != "spec"}
    assert comparability(
        [replace(reference, document=document), make_run("b"), make_run("c", spec={"account": {}})]
    ) == {"comparable": False, "reasons": ["COMPARE_ACCOUNT_DIFFERS"]}


def test_comparability_reasons_are_sorted_and_not_repeated() -> None:
    other = make_run("b", dataset_identity="other", evidence={"grade": "other"}, spec={"account": {}})
    other = replace(other, sessions=other.sessions[:-1])
    assert comparability(
        [make_run("a"), other, replace(other, document={**other.document, "run_id": "c"})]
    ) == {
        "comparable": False,
        "reasons": [
            "COMPARE_ACCOUNT_DIFFERS",
            "COMPARE_DATASET_DIFFERS",
            "COMPARE_GRADE_DIFFERS",
            "COMPARE_WINDOW_DIFFERS",
        ],
    }


@pytest.mark.parametrize("count", [0, 1])
def test_comparability_degenerate_sequences_are_vacuously_comparable(count: int) -> None:
    assert comparability([make_run("a")] * count) == {"comparable": True, "reasons": []}


def test_run_series_fields_are_frozen() -> None:
    run = make_run("a")
    with pytest.raises(FrozenInstanceError):
        run.sessions = ()


def test_compare_runs_has_exact_output_shape_and_preserves_metrics() -> None:
    reference, other = make_run("a"), make_run("b")
    result = compare_runs([reference, other], block=2, samples=8, seed=7)
    assert result == {
        "reference": "a",
        "comparable": True,
        "reasons": [],
        "runs": [
            {
                "run_id": run.document["run_id"],
                "strategy_id": "example",
                "configuration_hash": run.document["configuration_hash"],
                "dataset_identity": {"manifest_hash": "synthetic-example"},
                "grade": "verified",
                "metrics": run.document["metrics"],
            }
            for run in (reference, other)
        ],
        "differences": {"b": {"params": {}, "spec": {}}},
        "pairs": {
            "b": {
                "correlation": 1.0,
                "sharpe_difference": 0.0,
                "sharpe_difference_90": [0.0, 0.0],
                "total_return_difference": 0.0,
                "max_drawdown_difference": 0.0,
                "sessions": 4,
            }
        },
    }
    assert result["runs"][0]["metrics"] is reference.document["metrics"]


def test_compare_runs_annualizes_run_minus_reference_with_fixed_precision() -> None:
    # A +0.01 shift leaves the sample stdev sqrt(105)/600 unchanged.
    values = np.array([0.01, 0.02, -0.01, 0.03])
    reference = make_run("a", values)
    other = make_run("b", values + 0.01, metrics={"total_return": 0.173456789, "max_drawdown": 0.034567891})
    pair = compare_runs([reference, other], block=2, samples=8, seed=7)["pairs"]["b"]
    assert pair == {
        "correlation": 1.0,
        "sharpe_difference": round(6 / math.sqrt(105) * math.sqrt(252), 6),
        "sharpe_difference_90": [
            round((0.65 * math.sqrt(3) / 4 + 0.35 * 2 / math.sqrt(17)) * math.sqrt(252), 6),
            round((0.35 * 6 / math.sqrt(105) + 0.65 * math.sqrt(3)) * math.sqrt(252), 6),
        ],
        "total_return_difference": 0.12345679,
        "max_drawdown_difference": 0.02456789,
        "sessions": 4,
    }


def test_compare_runs_first_input_is_reference_and_order_is_preserved() -> None:
    values = np.array([0.01, 0.02, -0.01, 0.03])
    runs = [make_run("z", values + 0.01), make_run("a", values), make_run("m", values + 0.02)]
    result = compare_runs(runs, block=2)
    assert result["reference"] == "z"
    assert [run["run_id"] for run in result["runs"]] == ["z", "a", "m"]
    assert list(result["differences"]) == ["a", "m"]
    assert list(result["pairs"]) == ["a", "m"]
    assert result["pairs"]["a"]["sharpe_difference"] < 0 < result["pairs"]["m"]["sharpe_difference"]


def test_compare_runs_rounds_nonperfect_correlation_to_six_decimals() -> None:
    # Centered dot product is 5; squared norms are 5 and 10.
    reference = make_run("a", np.array([1.0, 2.0, 3.0, 4.0]))
    other = make_run("b", np.array([1.0, 4.0, 2.0, 5.0]))
    pair = compare_runs([reference, other], block=2)["pairs"]["b"]
    assert pair["correlation"] == 0.707107


def test_compare_runs_reports_reference_to_run_configuration_differences() -> None:
    other = make_run(
        "b",
        params={"lookback": 10},
        spec={"account": {"initial_cash": "100"}, "execution": {"costs": {"bps": "10"}}},
    )
    result = compare_runs([make_run("a"), other], block=2)
    assert result["comparable"] is True
    assert result["differences"] == {
        "b": {"params": {"lookback": [5, 10]}, "spec": {"execution.costs.bps": ["5", "10"]}}
    }


def test_compare_runs_incomparable_runs_have_no_pairs_but_keep_differences() -> None:
    other = make_run("b", dataset_identity="other", params={"lookback": 10})
    result = compare_runs([make_run("a"), other])
    assert result["comparable"] is False
    assert result["reasons"] == ["COMPARE_DATASET_DIFFERS"]
    assert result["pairs"] == {}
    assert result["differences"]["b"]["params"] == {"lookback": [5, 10]}


@pytest.mark.parametrize("count", [0, 1])
def test_compare_runs_needs_at_least_two_runs(count: int) -> None:
    with pytest.raises(ValueError, match="^COMPARE_NEEDS_TWO_RUNS$"):
        compare_runs([make_run("a")] * count)


def test_compare_runs_rejects_duplicate_ids_including_nonreference_runs() -> None:
    with pytest.raises(ValueError, match="^COMPARE_RUN_DUPLICATE:b$"):
        compare_runs([make_run("a"), make_run("b"), make_run("b")])


@pytest.mark.parametrize("constant_side", ["reference", "other", "both"])
def test_compare_runs_constant_returns_have_no_correlation(constant_side: str) -> None:
    variable = np.array([1.0, 2.0, 1.0, 2.0])
    flat = np.zeros(4)
    reference = make_run("a", flat if constant_side != "other" else variable)
    other = make_run("b", flat if constant_side != "reference" else variable)
    pair = compare_runs([reference, other], block=2)["pairs"]["b"]
    assert pair["correlation"] is None
    assert math.isfinite(pair["sharpe_difference"])


@pytest.mark.parametrize("count", [0, 1, 2])
def test_compare_runs_short_series_have_no_sharpe_or_interval(count: int) -> None:
    values = np.arange(count, dtype=float)
    pair = compare_runs([make_run("a", values), make_run("b", values)], block=1)["pairs"]["b"]
    assert pair["sharpe_difference"] is None
    assert pair["sharpe_difference_90"] is None
    assert pair["correlation"] == (1.0 if count == 2 else None)
    assert pair["sessions"] == count


def test_compare_runs_short_bootstrap_can_have_finite_point_estimate() -> None:
    pair = compare_runs([make_run("a"), make_run("b")])["pairs"]["b"]
    assert pair["sharpe_difference"] == 0.0
    assert pair["sharpe_difference_90"] is None


def test_compare_runs_three_sessions_are_enough_for_finite_sharpe() -> None:
    # Sample stdev is 0.01*sqrt(7/3); +0.01 changes annual Sharpe by sqrt(108).
    values = np.array([0.01, 0.02, -0.01])
    pair = compare_runs([make_run("a", values), make_run("b", values + 0.01)], block=1)["pairs"]["b"]
    assert pair["sharpe_difference"] == 10.392305
    assert pair["sharpe_difference_90"] is not None


def test_compare_runs_null_statistics_are_strictly_json_serializable() -> None:
    values = np.array([np.nan, np.inf, -np.inf])
    result = compare_runs([make_run("a", values), make_run("b", values)], block=1)
    assert result["pairs"]["b"] == {
        "correlation": None,
        "sharpe_difference": None,
        "sharpe_difference_90": None,
        "total_return_difference": 0.0,
        "max_drawdown_difference": 0.0,
        "sessions": 0,
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_compare_runs_filters_nonfinite_returns_jointly_for_all_pair_statistics() -> None:
    values = np.array([0.01, np.nan, 0.02, -0.01, 0.8, 0.03])
    other = np.array([0.02, 0.7, 0.03, 0.0, np.inf, 0.04])
    pair = compare_runs([make_run("a", values), make_run("b", other)], block=2, samples=8, seed=7)["pairs"][
        "b"
    ]
    clean = np.array([0.01, 0.02, -0.01, 0.03])
    expected = compare_runs([make_run("a", clean), make_run("b", clean + 0.01)], block=2, samples=8, seed=7)[
        "pairs"
    ]["b"]
    assert pair == expected
    assert pair["sessions"] == 4


@pytest.mark.parametrize("side", ["reference", "other", "both"])
@pytest.mark.parametrize("part", ["params", "spec"])
def test_compare_runs_missing_optional_metadata_has_empty_diff(side: str, part: str) -> None:
    reference, other = make_run("a"), make_run("b")
    if side != "other":
        reference = replace(
            reference, document={key: value for key, value in reference.document.items() if key != part}
        )
    if side != "reference":
        other = replace(other, document={key: value for key, value in other.document.items() if key != part})
    result = compare_runs([reference, other], block=2)
    assert result["comparable"] is True
    assert result["differences"]["b"][part] == {}


def test_compare_runs_is_json_serializable_and_deterministic_without_mutating_inputs() -> None:
    values = np.array([0.01, 0.02, -0.01, 0.03])
    reference, other = make_run("a", values), make_run("b", values + 0.01)
    before = json.dumps([reference.document, other.document], sort_keys=True)
    result = compare_runs([reference, other], block=2, samples=20, seed=19)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert compare_runs([reference, other], block=2, samples=20, seed=19) == result
    assert json.dumps([reference.document, other.document], sort_keys=True) == before
    np.testing.assert_array_equal(reference.returns, values)
    np.testing.assert_array_equal(other.returns, values + 0.01)
