# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fcntl
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from signalquarry._internal.contracts.factor_spec import FactorEvaluationSpecV1, FactorSpecV1
from signalquarry._internal.contracts.spec import HoldoutSpec
from signalquarry._internal.factors.expr import GRAMMAR_VERSION, parse_expression
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.factor_holdouts import (
    check_factor_training_window,
    effective_factor_declaration,
    factor_evidence_write,
    factor_family_seal,
    factor_holdout_start_for,
)
from signalquarry._internal.validation.factor_trials import (
    FactorTrialConfiguration,
    factor_trial_accounting,
    record_factor_trial,
    record_formula_trial,
    reserve_factor_trials,
)

AT = datetime(2026, 1, 6, tzinfo=UTC)


def _identity(character: str) -> str:
    return "sha256:" + character * 64


@pytest.fixture(params=["project", "per_family"])
def root(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    (tmp_path / "signalquarry.toml").write_text(f'[evidence]\nlayout = "{request.param}"\n')
    return tmp_path


def _bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and "evidence" in path.relative_to(root).parts
    }


def _seal(root: Path, family: str = "alpha", budget: int = 2, **changes: Any) -> dict[str, Any]:
    return ledger.append(
        root,
        "factor_holdouts",
        family,
        {
            "kind": "seal",
            "family": family,
            "trial_budget": budget,
            "holdout_start": "2026-01-06",
            "holdout": {"months": 12, "training_cutoff": None},
            "dataset_id": "test-dataset",
            "dataset_identity": _identity("b"),
            "dataset_last_session": "2026-01-09",
            **changes,
        },
    )


def _configuration(expression: str = "close") -> FactorTrialConfiguration:
    return FactorTrialConfiguration(
        factor_configuration_hash=parse_expression(expression).identity,
        dataset_identity=_identity("b"),
        universe_identity=_identity("c"),
        label_identity=_identity("d"),
        decision_sessions=(date(2025, 12, 1), date(2025, 12, 2)),
        evaluation=FactorEvaluationSpecV1(),
    )


def _formula(root: Path, expression: str = "close", **changes: Any) -> tuple[dict[str, Any], bool]:
    return record_formula_trial(
        root,
        **{
            "family": "alpha",
            "configuration": _configuration(expression),
            "expression": parse_expression(expression),
            "grammar_version": GRAMMAR_VERSION,
            "p_value": 0.125,
            "at": AT,
            "metrics": {"mean_ic": 0.25},
            **changes,
        },
    )


def _spec(identifier: str, budget: int, months: int, cutoff: date | None = None) -> FactorSpecV1:
    return FactorSpecV1(
        id=identifier,
        family="alpha",
        version="1",
        hypothesis={"statement": "Past prices predict returns.", "falsification": "IC is nonpositive."},
        evaluation=FactorEvaluationSpecV1(
            trial_budget=budget, holdout=HoldoutSpec(months=months, training_cutoff=cutoff)
        ),
    )


def test_factor_effective_declaration_uses_independent_conservative_extremes() -> None:
    specs = [
        (_spec("first", 9, 3), _identity("a")),
        (_spec("second", 20, 12, date(2024, 2, 29)), _identity("b")),
        (_spec("third", 5, 6, date(2024, 1, 31)), _identity("c")),
    ]
    declaration = effective_factor_declaration(specs)
    assert declaration["trial_budget"] == 5
    assert declaration["holdout"] == {"months": 12, "training_cutoff": "2024-01-31"}
    assert declaration["factors"] == [
        {
            "id": "first",
            "configuration_hash": _identity("a"),
            "trial_budget": 9,
            "holdout": {"months": 3, "training_cutoff": None},
        },
        {
            "id": "second",
            "configuration_hash": _identity("b"),
            "trial_budget": 20,
            "holdout": {"months": 12, "training_cutoff": "2024-02-29"},
        },
        {
            "id": "third",
            "configuration_hash": _identity("c"),
            "trial_budget": 5,
            "holdout": {"months": 6, "training_cutoff": "2024-01-31"},
        },
    ]


def test_factor_effective_declaration_without_cutoffs() -> None:
    assert effective_factor_declaration([(_spec("first", 9, 3), _identity("a"))])["holdout"] == {
        "months": 3,
        "training_cutoff": None,
    }


def test_factor_effective_declaration_refuses_empty_family() -> None:
    with pytest.raises(ledger.LedgerError, match="FACTOR_FAMILY_EMPTY"):
        effective_factor_declaration([])


@pytest.mark.parametrize(
    ("last", "months", "cutoff", "expected"),
    [
        (date(2024, 3, 31), 1, None, date(2024, 3, 1)),
        (date(2023, 3, 31), 1, None, date(2023, 3, 1)),
        (date(2024, 1, 31), 1, None, date(2024, 1, 1)),
        (date(2024, 3, 31), 1, date(2024, 2, 28), date(2024, 2, 29)),
        (date(2024, 3, 31), 1, date(2024, 2, 29), date(2024, 3, 1)),
        (date(2024, 3, 31), 1, date(2024, 3, 1), date(2024, 3, 1)),
        (date(2024, 3, 31), 0, date(2024, 2, 29), date(2024, 3, 1)),
        (date(2024, 3, 31), 0, None, None),
    ],
)
def test_factor_seal_start_month_end_and_cutoff_boundary(
    last: date, months: int, cutoff: date | None, expected: date | None
) -> None:
    assert factor_holdout_start_for(HoldoutSpec(months=months, training_cutoff=cutoff), last) == expected


def test_factor_accounting_uses_every_trial_identity_and_only_family_p_values(root: Path) -> None:
    _seal(root)
    _seal(root, "beta")
    record_factor_trial(
        root,
        family="alpha",
        factor_id="legacy",
        configuration=_configuration(),
        at=AT,
        metrics={"p_value": 0.1},
    )
    record_factor_trial(
        root,
        family="alpha",
        factor_id="legacy",
        configuration=replace(_configuration(), dataset_identity=_identity("e")),
        at=AT,
        metrics={},
    )
    record_factor_trial(
        root,
        family="beta",
        factor_id="legacy",
        configuration=_configuration(),
        at=AT,
        metrics={"p_value": 0.9},
    )
    alpha = factor_trial_accounting(root, "alpha")
    assert (alpha.family_trials_used, alpha.project_trials_used) == (2, 3)
    assert alpha.previous_family_p_values == (0.1,)
    assert alpha.trials_without_p_value == 1
    assert (alpha.effective_budget, alpha.remaining_budget) == (2, 0)
    beta = factor_trial_accounting(root, "beta")
    assert (beta.family_trials_used, beta.project_trials_used, beta.previous_family_p_values) == (
        1,
        3,
        (0.9,),
    )


def test_factor_accounting_does_not_consume_strategy_trials_or_extensions(root: Path) -> None:
    _seal(root)
    ledger.append(
        root, "trials", "alpha", {"kind": "trial", "family": "alpha", "configuration_hash": _identity("a")}
    )
    ledger.append(root, "trials", "alpha", {"kind": "budget_extension", "family": "alpha", "added": 1000})
    accounting = factor_trial_accounting(root, "alpha")
    assert (accounting.family_trials_used, accounting.project_trials_used, accounting.effective_budget) == (
        0,
        0,
        2,
    )


def test_factor_accounting_without_seal_refuses_without_writing(root: Path) -> None:
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_HOLDOUT_UNSEALED"):
        factor_trial_accounting(root, "alpha")
    assert _bytes(root) == before


def test_factor_budget_preflight_allows_exact_remaining_without_writing(root: Path) -> None:
    _seal(root)
    before = _bytes(root)
    assert reserve_factor_trials(root, "alpha", 2).remaining_budget == 2
    assert _bytes(root) == before


def test_factor_budget_preflight_refuses_one_over_remaining_without_writing(root: Path) -> None:
    _seal(root)
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_BUDGET_EXHAUSTED"):
        reserve_factor_trials(root, "alpha", 3)
    assert _bytes(root) == before


@pytest.mark.parametrize("requested", [-1, True, 1.5])
def test_factor_budget_preflight_rejects_invalid_requests(root: Path, requested: Any) -> None:
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_CONFIG_INVALID"):
        reserve_factor_trials(root, "alpha", requested)
    assert _bytes(root) == before


def test_factor_formula_trial_binds_expression_grammar_and_p_value(root: Path) -> None:
    seal = _seal(root)
    entry, new = _formula(
        root, "rank( close )", metrics={"grammar_version": -1, "p_value": 1, "expression": "forged"}
    )
    assert new
    assert entry["factor_configuration_hash"] == parse_expression("rank(close)").identity
    assert entry["trial_configuration_hash"] == _configuration("rank(close)").configuration_hash
    assert entry["metrics"] == {
        "expression": "rank(close)",
        "grammar_version": GRAMMAR_VERSION,
        "p_value": 0.125,
        "factor_holdout": {
            "seal_hash": seal["hash"],
            "holdout_start": "2026-01-06",
            "trial_budget": 2,
            "holdout": {"months": 12, "training_cutoff": None},
        },
    }
    assert ledger.trial_summary(root)["project_count"] == 0


def test_factor_formula_trial_is_idempotent(root: Path) -> None:
    _seal(root)
    first, _ = _formula(root)
    before = _bytes(root)
    repeated, new = _formula(root)
    assert not new and repeated == first
    assert _bytes(root) == before


def test_factor_formula_trial_unsealed_refusal_writes_nothing(root: Path) -> None:
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_HOLDOUT_UNSEALED"):
        _formula(root)
    assert _bytes(root) == before


def test_factor_formula_trial_at_budget_limit_is_allowed(root: Path) -> None:
    _seal(root, budget=1)
    assert _formula(root)[1]
    assert factor_trial_accounting(root, "alpha").remaining_budget == 0


def test_factor_formula_trial_over_budget_refusal_writes_nothing(root: Path) -> None:
    _seal(root, budget=1)
    _formula(root)
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_BUDGET_EXHAUSTED"):
        _formula(root, "open")
    assert _bytes(root) == before


def test_factor_formula_repeat_at_exhausted_budget_succeeds_without_writing(root: Path) -> None:
    _seal(root, budget=1)
    first, _ = _formula(root)
    before = _bytes(root)
    assert _formula(root) == (first, False)
    assert _bytes(root) == before


def test_factor_caller_evaluation_budget_cannot_enlarge_fixed_budget(root: Path) -> None:
    _seal(root, budget=1)
    enlarged = replace(_configuration(), evaluation=FactorEvaluationSpecV1(trial_budget=10000))
    entry, new = _formula(root, configuration=enlarged)
    assert new and entry["metrics"]["factor_holdout"]["trial_budget"] == 1
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_BUDGET_EXHAUSTED"):
        _formula(root, "open", configuration=replace(_configuration("open"), evaluation=enlarged.evaluation))
    assert _bytes(root) == before


def test_factor_budget_is_not_a_caller_parameter(root: Path) -> None:
    before = _bytes(root)
    with pytest.raises(TypeError, match="family_budget"):
        cast(Any, reserve_factor_trials)(root, "alpha", 1, family_budget=10000)
    with pytest.raises(TypeError, match="family_budget"):
        _formula(root, family_budget=10000)
    assert _bytes(root) == before


def test_factor_formula_identity_mismatch_refuses_without_writing(root: Path) -> None:
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_INPUT_UNVERIFIED"):
        _formula(root, configuration=_configuration("open"))
    assert _bytes(root) == before


@pytest.mark.parametrize("grammar_version", [0, True, 1.5])
def test_factor_formula_invalid_grammar_version_refuses_without_writing(
    root: Path, grammar_version: Any
) -> None:
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_INPUT_UNVERIFIED"):
        _formula(root, grammar_version=grammar_version)
    assert _bytes(root) == before


@pytest.mark.parametrize("p_value", [True, -0.1, 1.1, float("nan"), float("inf"), "bad", None, 10**400])
def test_factor_formula_invalid_p_value_refuses_without_writing(root: Path, p_value: Any) -> None:
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_INPUT_UNVERIFIED"):
        _formula(root, p_value=p_value)
    assert _bytes(root) == before


def test_factor_invalid_historical_p_value_fails_closed(root: Path) -> None:
    _seal(root)
    record_factor_trial(
        root,
        family="alpha",
        factor_id="legacy",
        configuration=_configuration(),
        at=AT,
        metrics={"p_value": "bad"},
    )
    before = _bytes(root)
    with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_INPUT_UNVERIFIED"):
        reserve_factor_trials(root, "alpha", 1)
    assert _bytes(root) == before


def test_factor_historical_null_p_value_is_counted_as_missing(root: Path) -> None:
    _seal(root)
    record_factor_trial(
        root,
        family="alpha",
        factor_id="legacy",
        configuration=_configuration(),
        at=AT,
        metrics={"p_value": None},
    )
    assert factor_trial_accounting(root, "alpha").trials_without_p_value == 1


def test_factor_accounting_legacy_overrun_has_zero_remaining(root: Path) -> None:
    _seal(root, budget=1)
    for expression in ("close", "open"):
        record_factor_trial(
            root,
            family="alpha",
            factor_id="legacy",
            configuration=_configuration(expression),
            at=AT,
            metrics={},
        )
    assert factor_trial_accounting(root, "alpha").remaining_budget == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"trial_budget": True},
        {"trial_budget": 0},
        {"kind": "opening"},
        {"holdout_start": "bad"},
        {"holdout": None},
    ],
)
def test_factor_malformed_seal_fails_closed(root: Path, changes: dict[str, Any]) -> None:
    _seal(root, **changes)
    with pytest.raises(ledger.LedgerError, match="EVIDENCE_LOG_CORRUPT"):
        factor_family_seal(root, "alpha")


def test_factor_duplicate_seal_fails_closed(root: Path) -> None:
    _seal(root)
    _seal(root)
    with pytest.raises(ledger.LedgerError, match="EVIDENCE_LOG_CORRUPT"):
        factor_family_seal(root, "alpha")


def test_factor_missing_seal_field_fails_closed(root: Path) -> None:
    ledger.append(root, "factor_holdouts", "alpha", {"family": "alpha", "kind": "seal"})
    with pytest.raises(ledger.LedgerError, match="EVIDENCE_LOG_CORRUPT"):
        factor_family_seal(root, "alpha")


def test_factor_family_slug_is_checked_before_reading(root: Path) -> None:
    with pytest.raises(ledger.LedgerError, match="USAGE_INVALID"):
        factor_family_seal(root, "../outside")


def test_factor_index_mismatch_blocks_accounting(tmp_path: Path) -> None:
    (tmp_path / "signalquarry.toml").write_text('[evidence]\nlayout = "per_family"\n')
    _seal(tmp_path)
    ledger.project_index(tmp_path).path.unlink()
    with pytest.raises(ledger.LedgerError, match="EVIDENCE_INDEX_MISMATCH"):
        factor_trial_accounting(tmp_path, "alpha")


def test_factor_writer_lock_refuses_busy_without_writing(root: Path) -> None:
    before = _bytes(root)
    with (root / "signalquarry.toml").open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        with pytest.raises(ledger.LedgerError, match="FACTOR_EVIDENCE_BUSY"), factor_evidence_write(root):
            pytest.fail("busy lock must not enter")
    assert _bytes(root) == before
    with factor_evidence_write(root):
        pass


CALENDAR = (date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8))


def test_factor_training_exact_longest_horizon_gap_is_allowed() -> None:
    check_factor_training_window(
        training_cutoff=CALENDAR[0], seal_start=CALENDAR[2], longest_horizon=2, sessions=CALENDAR
    )


def test_factor_training_one_session_short_is_refused() -> None:
    with pytest.raises(ledger.LedgerError, match="FACTOR_TRAINING_WINDOW_INVALID"):
        check_factor_training_window(
            training_cutoff=CALENDAR[1], seal_start=CALENDAR[2], longest_horizon=2, sessions=CALENDAR
        )


def test_factor_training_seal_on_non_session_uses_first_sealed_session() -> None:
    check_factor_training_window(
        training_cutoff=CALENDAR[0], seal_start=date(2026, 1, 4), longest_horizon=1, sessions=CALENDAR
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"training_cutoff": date(2026, 1, 4)},
        {"training_cutoff": date(2026, 1, 9)},
        {"training_cutoff": CALENDAR[2]},
        {"training_cutoff": CALENDAR[3]},
        {"seal_start": date(2026, 1, 9)},
        {"seal_start": CALENDAR[0]},
        {"sessions": ()},
        {"sessions": CALENDAR[::-1]},
        {"sessions": (CALENDAR[0], CALENDAR[0])},
        {"sessions": ("2026-01-02",)},
        {"longest_horizon": True},
        {"longest_horizon": 0},
        {"training_cutoff": "2026-01-02"},
        {"seal_start": "2026-01-06"},
    ],
)
def test_factor_training_invalid_calendars_and_cutoffs_are_refused(changes: dict[str, Any]) -> None:
    arguments = {
        "training_cutoff": CALENDAR[0],
        "seal_start": CALENDAR[2],
        "longest_horizon": 2,
        "sessions": CALENDAR,
        **changes,
    }
    with pytest.raises(ledger.LedgerError, match="FACTOR_TRAINING_WINDOW_INVALID"):
        check_factor_training_window(**arguments)
