# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from types import MappingProxyType

import numpy as np
import pytest

from signalquarry._internal.factors.evaluate import ScorePanel, SyntheticLabels
from signalquarry._internal.factors.expr import FactorExpression, _Node, evaluate_expression, parse_expression
from signalquarry._internal.factors.search import (
    FormulaSearchConfig,
    FormulaTrainingInput,
    _benjamini_hochberg,
    _crossover,
    _effective_observations,
    _expected_max_t,
    _mutate,
    _parse_root,
    _random_expression,
    _replace,
    _t_statistic,
    search_expressions,
)
from signalquarry.sdk.factors import FactorCtx


def _identity(character: str) -> str:
    return "sha256:" + character * 64


def _training_input(
    *, train_rows: int = 160, future_rows: int = 12, symbols_count: int = 24, planted: bool = True
) -> FormulaTrainingInput:
    rng = np.random.Generator(np.random.PCG64(112))
    total_rows = train_rows + future_rows
    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(total_rows))
    train_sessions = sessions[:train_rows]
    symbols = tuple(f"S{index:02}" for index in range(symbols_count))
    exposure = np.linspace(-1.0, 1.0, symbols_count, dtype=np.float64)
    if planted:
        close = 100 + np.arange(total_rows)[:, None] * 0.2 * exposure[None, :]
        close += rng.normal(0, 0.03, (total_rows, symbols_count))
        forward = np.maximum(exposure[None, :] + rng.normal(0, 0.3, (total_rows, symbols_count)), -0.95)
    else:
        close = 100 + np.cumsum(rng.normal(0, 0.5, (total_rows, symbols_count)), axis=0)
        forward = rng.normal(0, 0.02, (total_rows, symbols_count))
    ends: list[date | None] = [*sessions[1:], None]
    forward[-1] = np.nan
    panels = MappingProxyType(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 100_000 + rng.uniform(0, 50_000, (total_rows, symbols_count)),
        }
    )
    context = FactorCtx(
        decision_session=sessions[-1] + timedelta(days=1),
        sessions=train_sessions,
        universe=symbols,
        _panels=MappingProxyType({key: value[:train_rows] for key, value in panels.items()}),
    )
    eligible = np.ones((train_rows, symbols_count), dtype=np.bool_)
    eligible[::13, 0] = False
    labels = SyntheticLabels(
        dataset_identity=_identity("1"),
        label_identity=_identity("3"),
        sessions=sessions,
        symbols=symbols,
        forward_returns={1: forward},
        outcome_end_sessions={1: tuple(ends)},
    )
    return FormulaTrainingInput(
        dataset_identity=_identity("1"),
        universe_identity=_identity("2"),
        context=context,
        eligible=eligible,
        labels=labels,
    )


def _config(**updates: object) -> FormulaSearchConfig:
    return replace(
        FormulaSearchConfig(
            training_cutoff=date(2024, 1, 2) + timedelta(days=159),
            horizon=1,
            seed=17,
            budget=18,
            family_budget=40,
            population_size=8,
        ),
        **updates,
    )


def test_search_is_deterministic_and_finds_a_planted_training_signal() -> None:
    data = _training_input()
    config = _config()
    first = search_expressions(data, config)
    second = search_expressions(data, config)

    assert first == second
    assert first.scope == "synthetic"
    assert first.trials_evaluated == config.budget
    assert first.training_start == data.context.sessions[0]
    assert first.training_cutoff == data.context.sessions[-1]
    assert first.family_trials_after_batch == first.trials_evaluated
    assert first.trial_ledger_written is False
    assert first.holdout_accessed is False
    assert first.evidence_grade == "none"
    planted = next(item for item in first.candidates if item.expression == "rank(delta(close, 1))")
    assert planted.mean_ic is not None and planted.mean_ic > 0.8
    assert planted.discovery
    assert planted.identity in first.discoveries


def test_search_ignores_outcomes_that_end_after_training_cutoff() -> None:
    data = _training_input(train_rows=120, future_rows=24)
    original_returns = data.labels.forward_returns[1]
    changed_returns = np.empty(original_returns.shape, dtype=object)
    changed_returns[: len(data.context.sessions)] = original_returns[: len(data.context.sessions)]
    changed_returns[len(data.context.sessions) :] = "HELDOUT_MUST_NOT_BE_READ"
    changed_labels = replace(
        data.labels,
        label_identity=_identity("4"),
        forward_returns={1: changed_returns},
    )
    changed = replace(data, labels=changed_labels)
    config = _config(
        training_cutoff=data.context.sessions[-1], budget=12, family_budget=30, population_size=6
    )

    first = search_expressions(data, config)
    second = search_expressions(changed, config)
    assert first.training_label_identity == second.training_label_identity
    assert first.candidates == second.candidates
    assert first.discoveries == second.discoveries


def test_null_labels_do_not_produce_fdr_discoveries() -> None:
    data = _training_input(train_rows=180, future_rows=8, planted=False)
    report = search_expressions(
        data,
        _config(
            training_cutoff=data.context.sessions[-1],
            budget=32,
            family_budget=50,
            population_size=12,
            project_trials_used=20,
        ),
    )

    assert report.trials_evaluated == 32
    assert report.discoveries == ()
    assert all(not candidate.discovery for candidate in report.candidates)


def test_search_applies_library_redundancy_penalty() -> None:
    data = _training_input()
    expression = parse_expression("rank(close)")
    values = evaluate_expression(expression, data.context, eligible=data.eligible)
    accepted = ScorePanel(
        data.dataset_identity,
        data.universe_identity,
        data.context.sessions,
        data.context.universe,
        values,
        data.eligible,
    )
    report = search_expressions(
        data,
        _config(budget=8, family_budget=20, population_size=4, redundancy_penalty=1.0),
        accepted={"prior": accepted},
    )

    candidate = next(item for item in report.candidates if item.expression == "rank(close)")
    assert candidate.max_abs_library_correlation == pytest.approx(1.0)


def test_search_rejects_invalid_cutoff_labels_and_exhausted_budget() -> None:
    data = _training_input()
    with pytest.raises(ValueError, match="FACTOR_SEARCH_BUDGET_EXHAUSTED"):
        search_expressions(
            data,
            _config(budget=2, family_budget=3, family_trials_used=2, project_trials_used=2),
        )

    no_end_sessions = replace(data.labels, outcome_end_sessions=None)
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(replace(data, labels=no_end_sessions), _config())

    wrong_cutoff = _config(training_cutoff=data.context.sessions[-2])
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(data, wrong_cutoff)


def test_search_uses_prior_family_p_values_in_bh_adjustment() -> None:
    data = _training_input()
    report = search_expressions(
        data,
        _config(
            budget=8,
            family_budget=20,
            family_trials_used=2,
            project_trials_used=2,
            previous_family_p_values=(0.01, 0.2),
            population_size=4,
        ),
    )
    assert all(0 <= candidate.q_value <= 1 for candidate in report.candidates)


def test_genetic_tree_operators_return_valid_safe_expressions() -> None:
    left = parse_expression("rank(ts_corr(close, volume, 5) + delta(open, 2))")
    right = parse_expression("zscore(ts_mean(high, 10) / close)")
    mutations = [_mutate(left, np.random.Generator(np.random.PCG64(seed))) for seed in range(50)]
    crossovers = [_crossover(left, right, np.random.Generator(np.random.PCG64(seed))) for seed in range(20)]
    randoms = [_random_expression(np.random.Generator(np.random.PCG64(seed))) for seed in range(30)]

    for candidate in (*mutations, *crossovers, *randoms):
        if candidate is not None:
            assert parse_expression(candidate.canonical).identity == candidate.identity


def test_search_rejects_invalid_config_and_provenance() -> None:
    data = _training_input()
    with pytest.raises(ValueError, match="FACTOR_SEARCH_CONFIG_INVALID"):
        search_expressions(data, replace(_config(), previous_family_p_values=[0.1]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="FACTOR_SEARCH_CONFIG_INVALID"):
        search_expressions(data, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(None, _config())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(replace(data, dataset_identity=_identity("9")), _config())

    invalid_context = replace(data.context, decision_session="not-a-date")
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(replace(data, context=invalid_context), _config())  # type: ignore[arg-type]


def test_search_rejects_invalid_eligibility_and_panels() -> None:
    data = _training_input()
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(replace(data, eligible=data.eligible.astype(np.int8)), _config())
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(replace(data, eligible=data.eligible[:-1]), _config())

    panels = dict(data.context._panels)
    panels.pop("high")
    missing_panel = replace(data.context, _panels=MappingProxyType(panels))
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(replace(data, context=missing_panel), _config())

    panels = dict(data.context._panels)
    broken_close = np.array(panels["close"], copy=True)
    broken_close[0, 0] = np.inf
    panels["close"] = broken_close
    infinite_panel = replace(data.context, _panels=MappingProxyType(panels))
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(replace(data, context=infinite_panel), _config())


def test_search_rejects_malformed_forward_labels_and_end_dates() -> None:
    data = _training_input()
    returns = np.array(data.labels.forward_returns[1], copy=True)
    ends = list(data.labels.outcome_end_sessions[1])
    invalid_labels = (
        replace(data.labels, forward_returns={1: returns[:, :-1]}),
        replace(
            data.labels, forward_returns={1: np.where(np.indices(returns.shape)[0] == 0, np.inf, returns)}
        ),
        replace(
            data.labels, forward_returns={1: np.where(np.indices(returns.shape)[0] == 0, -1.01, returns)}
        ),
        replace(data.labels, outcome_end_sessions={1: tuple(ends[:-1])}),
    )
    for labels in invalid_labels:
        with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
            search_expressions(replace(data, labels=labels), _config())

    ends[0] = data.context.sessions[0]
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(
            replace(data, labels=replace(data.labels, outcome_end_sessions={1: tuple(ends)})),
            _config(),
        )

    ends = list(data.labels.outcome_end_sessions[1])
    ends[0] = None
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(
            replace(data, labels=replace(data.labels, outcome_end_sessions={1: tuple(ends)})),
            _config(),
        )

    bad_adv = np.ones((1, len(data.context.universe)), dtype=np.float64)
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(
            replace(data, labels=replace(data.labels, predecision_adv=bad_adv)),
            _config(),
        )


def test_search_accepts_null_terminal_label_and_valid_adv() -> None:
    data = _training_input(train_rows=60, future_rows=0)
    labels = replace(
        data.labels,
        predecision_adv=np.full(data.labels.forward_returns[1].shape, 100_000.0),
    )
    report = search_expressions(
        replace(data, labels=labels),
        _config(training_cutoff=data.context.sessions[-1], budget=1, population_size=2),
    )
    assert report.trials_evaluated == 1


def test_search_helpers_cover_degenerate_samples_and_invalid_trees() -> None:
    assert _effective_observations(np.array([0.1, 0.2])) == 2.0
    assert _effective_observations(np.ones(10)) == 10.0
    count, effective, icir, statistic = _t_statistic(tuple([0.0] * 30))
    assert (count, effective, icir, statistic) == (30, 30.0, None, None)
    assert _expected_max_t(1) == 0.0
    assert _benjamini_hochberg(()) == ()

    leaf = _Node("panel", ("close",))
    assert _replace(leaf, (0,), _Node("panel", ("open",))) is leaf
    assert _parse_root(_Node("unsupported", ())) is None

    malformed = FactorExpression("unknown(x)", "invalid", _Node("unknown", ("x",)))
    _mutate(malformed, np.random.Generator(np.random.PCG64(0)))


def test_search_rejects_invalid_accepted_factor_library() -> None:
    data = _training_input()
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(data, _config(), accepted=[])  # type: ignore[arg-type]

    values = np.zeros(data.eligible.shape, dtype=np.float64)
    wrong_dataset = ScorePanel(
        _identity("9"),
        data.universe_identity,
        data.context.sessions,
        data.context.universe,
        values,
        data.eligible,
    )
    with pytest.raises(ValueError, match="FACTOR_SEARCH_INPUT_INVALID"):
        search_expressions(data, _config(), accepted={"wrong-dataset": wrong_dataset})
