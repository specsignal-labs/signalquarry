# SPDX-License-Identifier: Apache-2.0
"""The search report recomputed from its own candidates with an independent reference."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from signalquarry._internal.factors.evaluate import ScorePanel
from signalquarry._internal.factors.expr import evaluate_expression, parse_expression
from signalquarry._internal.factors.search import FormulaSearchReport, search_expressions

from .test_factor_search import _config, _training_input

_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def _haircut(trials: int) -> float:
    if trials <= 1:
        return 0.0
    return (1 - _GAMMA) * _NORMAL.inv_cdf(1 - 1 / trials) + _GAMMA * _NORMAL.inv_cdf(
        1 - 1 / (trials * math.e)
    )


def _reference_q_values(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * count
    running = 1.0
    for position in range(count, 0, -1):
        index = order[position - 1]
        running = min(running, p_values[index] * count / position)
        adjusted[index] = min(1.0, running)
    return adjusted


def _check_report(
    report: FormulaSearchReport, *, family_used: int, project_used: int, previous: tuple[float, ...]
) -> None:
    candidates = list(report.candidates)
    count = len(candidates)
    haircut = _haircut(project_used + count)
    # Candidates are sorted by fitness, so recover generation-independent values from each row.
    assert report.trials_evaluated == count
    assert report.family_trials_after_batch == family_used + count
    assert report.family_trials_used_before == family_used
    assert report.project_trials_used_before == project_used

    p_by_identity = {}
    for item in candidates:
        if item.t_statistic is None:
            assert item.deflated_t_statistic is None
            assert item.p_value == 1.0
            assert item.fitness == -1_000_000.0
        else:
            deflated = item.t_statistic - haircut
            assert item.deflated_t_statistic == pytest.approx(deflated, rel=1e-12, abs=1e-12)
            assert item.p_value == pytest.approx(_NORMAL.cdf(-deflated), rel=1e-12, abs=1e-15)
            expected = deflated - 0.01 * max(0, item.complexity - 1) - 0.25 * item.max_abs_library_correlation
            assert item.fitness == pytest.approx(expected, rel=1e-12, abs=1e-12)
        p_by_identity[item.identity] = item.p_value

    # q-values come from one Benjamini-Hochberg pass over: unseen earlier trials (p = 1), the
    # supplied earlier p-values, and this batch. Batch order is the order of evaluation, which
    # the report does not keep; the multiset of adjusted values is order independent.
    padding = [1.0] * (family_used - len(previous))
    combined = _reference_q_values([*padding, *previous, *p_by_identity.values()])
    expected_q = sorted(combined[len(padding) + len(previous) :])
    # q for the same p can differ with evaluation order only by position among exact ties, which
    # leaves the sorted multiset unchanged.
    assert sorted(item.q_value for item in candidates) == pytest.approx(expected_q, rel=1e-12, abs=1e-15)

    for item in candidates:
        assert 0.0 <= item.q_value <= 1.0
        assert item.discovery == (item.mean_ic is not None and item.mean_ic > 0 and item.q_value <= 0.05)
        assert item.q_value >= item.p_value - 1e-15

    keys = [(-item.fitness, item.identity) for item in candidates]
    assert keys == sorted(keys)
    assert report.discoveries == tuple(item.identity for item in candidates if item.discovery)


def test_first_batch_report_matches_the_independent_reference() -> None:
    data = _training_input()
    report = search_expressions(data, _config())
    _check_report(report, family_used=0, project_used=0, previous=())
    assert any(item.discovery for item in report.candidates)
    assert report.dataset_identity == data.dataset_identity
    assert report.universe_identity == data.universe_identity
    assert report.training_label_identity.startswith("sha256:")
    assert report.training_label_identity != data.labels.label_identity
    assert report.training_start == data.context.sessions[0]
    assert report.training_cutoff == data.context.sessions[-1]
    assert (report.horizon, report.seed, report.budget) == (1, 17, 18)
    assert report.scope == "synthetic"
    assert report.evidence_grade == "none"
    assert report.trial_ledger_written is False
    assert report.holdout_accessed is False


def test_later_batch_report_deflates_by_every_project_trial_and_pads_unseen_family_trials() -> None:
    data = _training_input()
    previous = (0.01, 0.5, 0.2)
    config = _config(
        family_trials_used=5,
        project_trials_used=40,
        previous_family_p_values=previous,
        budget=20,
        family_budget=60,
    )
    report = search_expressions(data, config)
    _check_report(report, family_used=5, project_used=40, previous=previous)
    first = search_expressions(data, _config(budget=20, family_budget=60))
    # More earlier trials raise the bar: every shared expression is deflated by a larger haircut.
    lookup = {item.identity: item for item in first.candidates}
    shared = [item for item in report.candidates if item.identity in lookup and item.t_statistic is not None]
    assert shared
    for item in shared:
        assert item.deflated_t_statistic is not None
        assert lookup[item.identity].deflated_t_statistic is not None
        assert item.deflated_t_statistic < lookup[item.identity].deflated_t_statistic  # type: ignore[operator]
        assert item.p_value >= lookup[item.identity].p_value


def test_effective_observations_are_reported_to_six_decimals() -> None:
    report = search_expressions(_training_input(), _config())
    for item in report.candidates:
        assert item.effective_observations == round(item.effective_observations, 6)
        assert 1.0 <= item.effective_observations <= item.observations


def test_a_candidate_without_a_defined_t_statistic_is_never_a_discovery() -> None:
    data = _training_input(train_rows=25, future_rows=4)
    report = search_expressions(data, _config(training_cutoff=data.context.sessions[-1], budget=6))
    assert report.candidates
    assert all(item.t_statistic is None for item in report.candidates)
    assert all(item.p_value == 1.0 and not item.discovery for item in report.candidates)
    assert report.discoveries == ()
    _check_report(report, family_used=0, project_used=0, previous=())


def test_library_correlation_is_reported_and_lowers_fitness() -> None:
    data = _training_input()
    plain = search_expressions(data, _config())
    context = data.context
    values = evaluate_expression(parse_expression("rank(delta(close, 1))"), context, eligible=data.eligible)
    library = {
        "planted": ScorePanel(
            data.dataset_identity,
            data.universe_identity,
            context.sessions,
            context.universe,
            values,
            data.eligible,
        )
    }
    with_library = search_expressions(data, _config(), accepted=library)
    assert plain.trials_evaluated == with_library.trials_evaluated
    by_identity = {item.identity: item for item in plain.candidates}
    assert all(item.max_abs_library_correlation == 0.0 for item in plain.candidates)
    twin = next(item for item in with_library.candidates if item.expression == "rank(delta(close, 1))")
    assert twin.max_abs_library_correlation > 0.99
    before = by_identity[twin.identity]
    assert twin.fitness == pytest.approx(before.fitness - 0.25 * twin.max_abs_library_correlation, rel=1e-9)
    assert all(0.0 <= item.max_abs_library_correlation <= 1.0 for item in with_library.candidates)


def test_the_search_is_reproducible_but_depends_on_the_seed() -> None:
    data = _training_input()
    a = search_expressions(data, _config())
    b = search_expressions(data, _config())
    c = search_expressions(data, _config(seed=18))
    assert a == b
    assert {item.identity for item in a.candidates} != {item.identity for item in c.candidates}
    assert np.isfinite([item.fitness for item in a.candidates]).all()
